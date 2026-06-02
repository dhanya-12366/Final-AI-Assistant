import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(
    BASE_DIR,
    "..",
    "arxiv_data",
    "arxiv-metadata-oai-snapshot.json"
)

output_file = os.path.join(
    BASE_DIR,
    "..",
    "arxiv_data",
    "filtered_cs_papers.json"
)

print("Reading from:")
print(input_file)

count = 0
max_papers = 2000

cs_categories = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.RO"
]

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:

        try:
            paper = json.loads(line)

            categories = paper.get("categories", "")

            if any(cat in categories for cat in cs_categories):

                record = {
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "categories": categories
                }

                outfile.write(
                    json.dumps(record) + "\n"
                )

                count += 1

                if count % 100 == 0:
                    print(f"{count} papers extracted")

                if count >= max_papers:
                    break

        except Exception:
            continue

print(f"\nDone. {count} papers extracted.")
print("Saved to:")
print(output_file)