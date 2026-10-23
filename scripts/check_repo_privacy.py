"""Verifica se algum arquivo privado esta (ou ja esteve) no Git. Anexo B do enunciado."""
import re
import subprocess
import sys

FORBIDDEN = [
    r"^data/private/", r"^data/teste_local/", r"^faces/", r"^embeddings/",
    r"^checkpoints/private/", r"^consent/",
    r"\.(db|sqlite|sqlite3|pt|pth|ckpt|onnx|safetensors)$", r"(^|/)weights\.bin$",
]


def git_lines(args):
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def matches(paths):
    return sorted(p for p in paths if any(re.search(rx, p) for rx in FORBIDDEN))


def main():
    atuais = matches(git_lines(["ls-files"]))
    historico = matches(git_lines(["log", "--all", "--name-only", "--pretty=format:"]))
    if atuais:
        print("ERRO - arquivos privados versionados agora:", *atuais, sep="\n  ")
    if historico:
        print("ERRO - arquivos privados presentes no HISTORICO:", *historico, sep="\n  ")
    if atuais or historico:
        sys.exit(1)
    print("OK - nenhum arquivo privado no repositorio nem no historico.")


if __name__ == "__main__":
    main()
