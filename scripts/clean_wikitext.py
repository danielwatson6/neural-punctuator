"""Script to clean the Wikitext data and segment it to the sentence-level.

The following special word-level tokens are introduced:
    <pad>: used for batch processing of variable-length sentences
    <unk>: unknown, out-of-vocabulary tokens
    <sos>: start-of-sentence token
    <eos>: end-of-sentence token
    <dash>: dash inside a compound word
    <num>: a number
    <num_dot>: a decimal separator inside a number
    <num_comma>: a comma separator inside a number

"""

from collections import Counter
import os
import re


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<sos>",
    "<eos>",
    "<dash>",
    "<num>",
    "<num_dot>",
    "<num_comma>",
]


def write_tokens(counts, wf):
    for token in SPECIAL_TOKENS:
        c = 0
        if token in counts:
            c = counts[token]
            del counts[token]
        wf.write(f"{token}\t{c}\n")

    for token, count in counts.most_common():
        wf.write(f"{token}\t{count}\n")


if __name__ == "__main__":

    for corpus, extension in [("2", "tokens"), ("103-raw", "raw")]:
        for partition in ["train", "valid", "test"]:
            root = os.path.join("data", f"wikitext-{corpus}")
            print(f"wikitext-{corpus} ({partition})\n  cleaning data...")

            rf = open(os.path.join(root, f"wiki.{partition}.{extension}"))
            wf = open(os.path.join(root, f"wiki.{partition}.clean"), "w")

            word_counts = None
            char_counts = None
            if partition == "train":
                word_counts = Counter()
                char_counts = Counter()

            for line in rf:

                # Ignore wikipedia headers.
                if line == "" or line.startswith(" ="):
                    continue

                line = re.sub(r"@-@", " <dash> ", line)
                line = re.sub(r"[0-9]+", " <num> ", line)
                line = re.sub(r"@\.@", " <num_dot> ", line)
                line = re.sub(r"@,@", " <num_comma> ", line)

                # Shrink spaces.
                line = re.sub(r"\s+", " ", line).strip()

                # Segment into sentences / independent clauses by tokenised '.', ';'.
                lines = []
                line_buf = []
                for word in line.split():
                    line_buf.append(word)

                    # Check for end of "sentence".
                    if word in [".", ";"]:
                        lines.append(" ".join(line_buf))
                        line_buf = []

                    if partition == "train":
                        word_counts[word] += 1

                        # Special tokens should not be segmented into characters.
                        if word in SPECIAL_TOKENS:
                            char_counts[word] += 1
                        else:
                            for char in word:
                                char_counts[char] += 1

                for line in lines:
                    wf.write(line + "\n")

            rf.close()
            wf.close()

            if partition == "train":
                print("  sorting train vocabulary...")
                with open(os.path.join(root, f"wiki.vocabulary.tsv"), "w") as f:
                    write_tokens(word_counts, f)

                print("  sorting train alphabet...")
                with open(os.path.join(root, "wiki.alphabet.tsv"), "w") as f:
                    write_tokens(char_counts, f)
