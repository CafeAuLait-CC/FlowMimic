"""Convert pure AIST++ captions into HumanML3D-style text token files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_spacy(model: str, package_dir: str | None):
    if package_dir:
        sys.path.insert(0, package_dir)

    import spacy

    return spacy.load(model)


def process_text(nlp, sentence: str) -> str:
    sentence = sentence.replace("-", "")
    doc = nlp(sentence)
    words: list[str] = []
    poses: list[str] = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if token.pos_ in {"NOUN", "VERB"} and word != "left":
            words.append(token.lemma_)
        else:
            words.append(word)
        poses.append(token.pos_)
    return " ".join(f"{word}/{pos}" for word, pos in zip(words, poses))


def read_captions(path: Path) -> list[str]:
    captions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        caption = line.strip()
        if caption:
            captions.append(caption.replace("#", " "))
    return captions


def convert_file(nlp, path: Path) -> tuple[list[str], list[str], list[str]]:
    captions = read_captions(path)
    tokens = [process_text(nlp, caption) for caption in captions]
    combined = [
        f"{caption}#{token_line}#0.0#0.0"
        for caption, token_line in zip(captions, tokens)
    ]
    return captions, tokens, combined


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create motion-latent-diffusion and stickmotion style "
            "HumanML3D text token files for AIST++ captions."
        )
    )
    parser.add_argument("--input-dir", default="data/AIST++/Texts")
    parser.add_argument("--mld-output-dir", default="data/AIST++/TextTokens/mld_texts")
    parser.add_argument(
        "--stick-text-output-dir", default="data/AIST++/TextTokens/stickmotion/texts"
    )
    parser.add_argument(
        "--stick-token-output-dir", default="data/AIST++/TextTokens/stickmotion/tokens"
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument(
        "--spacy-package-dir",
        default=None,
        help="Optional directory added to sys.path before loading the spaCy model.",
    )
    parser.add_argument("--file", default=None, help="Only process this input filename.")
    parser.add_argument(
        "--text",
        default=None,
        help="Tokenize one caption and print its HumanML3D token string.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print converted content and do not write output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nlp = load_spacy(args.spacy_model, args.spacy_package_dir)

    if args.text is not None:
        print(process_text(nlp, args.text.replace("#", " ").strip()))
        return

    input_dir = Path(args.input_dir)
    if args.file:
        paths = [input_dir / args.file]
    else:
        paths = sorted(input_dir.glob("*.txt"))
    if args.limit is not None:
        paths = paths[: args.limit]

    if not paths:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    for path in paths:
        captions, tokens, combined = convert_file(nlp, path)
        if args.preview:
            print(f"FILE: {path.name}")
            print("\n[MLD combined]")
            print("\n".join(combined))
            print("\n[stickmotion texts]")
            print("\n".join(captions))
            print("\n[stickmotion tokens]")
            print("\n".join(tokens))
            continue

        write_lines(Path(args.mld_output_dir) / path.name, combined)
        write_lines(Path(args.stick_text_output_dir) / path.name, captions)
        write_lines(Path(args.stick_token_output_dir) / path.name, tokens)


if __name__ == "__main__":
    main()
