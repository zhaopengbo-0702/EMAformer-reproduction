"""Expand the trusted upstream scripts without executing a shell."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

SEEDS = [2023, 2024, 2025, 2026, 2027, 2028]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expand(text, source="<string>"):
    """Strict literal assignments and nested for loops only; unknown syntax fails."""
    logical = re.sub(r"\\\r?\n", " ", text)
    lines = []
    for number, line in enumerate(logical.splitlines(), 1):
        tokens = shlex.split(line, comments=True, posix=True)
        if tokens:
            lines.append((number, tokens))

    def substitute(token, env):
        def repl(match):
            key = match.group(1) or match.group(2)
            if key not in env:
                raise ValueError(f"{source}: undefined variable {key}")
            return env[key]
        return re.sub(r"\$\{(\w+)\}|\$(\w+)", repl, token)

    def block(start, env, stop=False):
        output, index = [], start
        while index < len(lines):
            number, tokens = lines[index]
            if tokens == ["done"]:
                if not stop:
                    raise ValueError(f"{source}:{number}: unmatched done")
                return output, index + 1
            if tokens[0] == "for":
                if len(tokens) < 4 or tokens[2] != "in":
                    raise ValueError(f"{source}:{number}: unsupported loop")
                if index + 1 >= len(lines) or lines[index + 1][1] != ["do"]:
                    raise ValueError(f"{source}:{number}: expected do")
                end = None
                for value in tokens[3:]:
                    child = dict(env, **{tokens[1]: substitute(value, env)})
                    expanded, end = block(index + 2, child, True)
                    output.extend(expanded)
                index = end
                continue
            if len(tokens) == 1 and re.fullmatch(r"[A-Za-z_]\w*=.*", tokens[0]):
                name, value = tokens[0].split("=", 1)
                env[name] = substitute(value, env)
            elif tokens[0] in ("python", "python3"):
                command = [substitute(t, env) for t in tokens]
                if command[1:3] == ["-u", "run.py"]:
                    argv = command[3:]
                elif command[1:2] == ["run.py"]:
                    argv = command[2:]
                else:
                    raise ValueError(f"{source}:{number}: unexpected Python command")
                output.append({"argv": argv, "source": source, "logical_line": number})
            else:
                raise ValueError(f"{source}:{number}: unsupported syntax: {tokens}")
            index += 1
        if stop:
            raise ValueError(f"{source}: unterminated loop")
        return output, index

    return block(0, {})[0]


def options(argv):
    result = {}
    index = 0
    while index < len(argv):
        key = argv[index]
        if not key.startswith("--"):
            raise ValueError(f"Expected option, found {key}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[key[2:]] = argv[index + 1]
            index += 2
        else:
            result[key[2:]] = True
            index += 1
    return result


def replace_option(argv, key, value):
    argv = list(argv)
    flag = "--" + key
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            argv[index + 1] = str(value)
        else:
            raise ValueError(f"Cannot assign value to flag {flag}")
    else:
        argv.extend([flag, str(value)])
    return argv


def make_manifest(repo, main_seeds=SEEDS, extra_seeds=(2023,)):
    repo = Path(repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    runs, counts = [], {}
    for script in sorted((repo / "scripts").rglob("*.sh")):
        rel = script.relative_to(repo).as_posix()
        source_group = script.relative_to(repo / "scripts").parts[0]
        commands = expand(script.read_text(encoding="utf-8-sig"), rel)
        counts[source_group] = counts.get(source_group, 0) + len(commands)
        for command in commands:
            opts = options(command["argv"])
            group = ("upstream_extra_exchange" if source_group == "multivariate_forecasting"
                     and opts["model_id"].startswith("Exchange_") else source_group)
            for seed in main_seeds if group == "multivariate_forecasting" else extra_seeds:
                run = {"group": group, "seed": seed, "argv": command["argv"],
                       "extensions": {}, "source": command["source"],
                       "logical_line": command["logical_line"], "provenance": "upstream_active_script"}
                signature = digest({"argv": run["argv"], "seed": seed, "extensions": {}})
                run["id"] = f"{group}-{opts['model_id']}-{opts['model']}-s{seed}-{signature[:10]}"
                run["signature"] = signature
                runs.append(run)
    runs.sort(key=lambda run: (run["group"], run["seed"], run["source"], run["logical_line"]))
    return {"schema": 1, "upstream_commit": commit, "source_script_command_counts": counts,
            "resource_policy": {"max_cloud_tasks": 1, "max_gpus": 1, "max_parallel_training": 1},
            "seed_provenance": "Paper states six seeds, identities not published. Seeds 2023..2028 are explicit reproduction choices; upstream default is 2023.",
            "scope": "Only active released script configurations. Not the full paper matrix; reconstructed experiments are tracked separately.",
            "runs": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = make_manifest(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(data['runs']), "script_commands": data['source_script_command_counts']}))


if __name__ == "__main__":
    main()
