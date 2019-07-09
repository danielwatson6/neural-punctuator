from collections import Counter
import os
import re


if __name__ == "__main__":

    for corpus, extension in [("2", "tokens"), ("103-raw", "raw")]:
        for partition in ["train", "valid", "test"]:
            root = os.path.join("data", f"wikitext-{corpus}")
            print(f"wikitext-{corpus} ({partition})\n  cleaning data...")

            rf = open(os.path.join(root, f"wiki.{partition}.{extension}"))
            wf_inputs = open(os.path.join(root, f"wiki.{partition}.inputs"), "w")
            wf_labels = open(os.path.join(root, f"wiki.{partition}.labels"), "w")

            counts = None
            if partition == "train":
                counts = Counter()

            for line in rf:
                # Ignore wikipedia headers.
                line = line.strip()
                if line == "" or line.startswith(" ="):
                    continue

                line = re.sub(r"[0-9]+", "<num>", line)
                line = re.sub(r"@-@", "<dash>", line)
                line = re.sub(r"@\.@", "<num_dot>", line)
                line = re.sub(r"@,@", "<num_comma>", line)

                # Segment into sentences / independent clauses by tokenised '.', ';'.
                lines = []
                line_buf = []
                for token in line.split():

                    # Check for end of "sentence".
                    if token in [".", ";"]:
                        line_buf.append(token)
                        lines.append(" ".join(line_buf))
                        line_buf = []
                    else:
                        line_buf.append(token)

                    if partition == "train":
                        if token not in counts:
                            counts[token] = 0
                        counts[token] += 1

                for line in lines:
                    no_punctuation = re.sub(r"[^A-Za-z'<>_]", " ", line)
                    no_punctuation = re.sub(r" +", " ", no_punctuation).strip()
                    wf_inputs.write(no_punctuation + "\n")
                    wf_labels.write(line + "\n")

            # Write tokens sorted by frequency.
            if partition == "train":
                print("  sorting vocabulary...")
                with open(os.path.join(root, f"wiki.vocab.tsv"), "w") as wf_voc:

                    # Special tokens first!
                    for token in ["<pad>", "<sos>", "<eos>"]:
                        wf_voc.write(f"{token}\t0\n")

                    for token, count in counts.most_common():
                        wf_voc.write(f"{token}\t{count}\n")

            rf.close()
            wf_inputs.close()
            wf_labels.close()
