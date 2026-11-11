import argparse
import datetime
import json
import os
import re
import requests
import yaml

from bs4 import BeautifulSoup
from easydict import EasyDict as edict
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser
from tqdm import tqdm

from hallupeer.utils import load_configs


def parse_pdf_with_grobid(
    pdf_path: list[str],
    parse_log_dir: str = None,
    grobid_url: str = "http://localhost:8070",
):

    # If parse_log_dir is provided, ensure the directory exists
    if parse_log_dir:
        os.makedirs(parse_log_dir, exist_ok=True)

    for paper_dir in pdf_path:
        # Initialize independent failure log for each paper_dir
        failed_papers = []

        # Use parent directory name (e.g., NeurIPS2022, ICLR2023)
        if parse_log_dir:
            parent_dir_name = os.path.basename(os.path.dirname(paper_dir))
            log_file_path = os.path.join(
                parse_log_dir,
                f"parse_failures_{parent_dir_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

        # paper_dir now directly points to the directory containing PDFs (meta_dir + paper or meta_dir + pdfs)
        # Get parent directory as meta_dir
        meta_dir = os.path.dirname(paper_dir)

        # Create parsed_pdfs directory in meta_dir
        parsed_pdfs_dir = os.path.join(meta_dir, "parsed_pdfs")
        os.makedirs(parsed_pdfs_dir, exist_ok=True)

        # Recursively search for all PDF files
        pdf_files = []

        for root, dirs, files in os.walk(paper_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))

        for f in tqdm(
            pdf_files,
            total=len(pdf_files),
            desc="🔥 Processing PDFs in " + paper_dir,
        ):
            try:
                # 1️⃣ First extract abstract from header
                response_xml = requests.post(
                    f"{grobid_url}/api/processFulltextDocument",
                    files={"input": open(f, "rb")},
                ).content.decode("utf-8", errors="ignore")

                # Extract abstract from XML
                soup = BeautifulSoup(response_xml, "lxml-xml")

                title_elem = soup.find("title")
                abstract_elem = soup.find("abstract")
                if title_elem:
                    title_text = title_elem.get_text(separator=" ", strip=True)
                    tqdm.write(f"✅ Title found: {title_text}")
                else:
                    tqdm.write(f"❌ No title found in XML response")
                    title_text = None
                if abstract_elem:
                    abstract_text = abstract_elem.get_text(separator=" ", strip=True)
                    if "Fig" in abstract_text or "Table" in abstract_text:
                        # Avoid capturing figure/table captions as abstract
                        tqdm.write(
                            f"❌ Abstract seems to contain figure/table references. Ignoring."
                        )
                        abstract_text = None
                    else:
                        tqdm.write(f"✅ Abstract found via processFulltextDocument")
                else:
                    tqdm.write(f"❌ No abstract found in XML response")
                    abstract_text = None

                # 2️⃣ Use GrobidParser to extract full text
                loader = GenericLoader.from_filesystem(
                    path=f,
                    suffixes=[".pdf"],
                    parser=GrobidParser(
                        segment_sentences=True,
                    ),  # Specify single file  # Use Grobid to parse PDF
                )

                docs = loader.load()

                # Check if docs is empty
                if not docs:
                    tqdm.write(f"❗No content extracted from {f}. Skipping.")
                    error_info = {
                        "file_path": f,
                        "error_type": "EmptyContent",
                        "error_message": "No content extracted from PDF",
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    failed_papers.append(error_info)
                    continue

                print(f"📄 Extracted {len(docs)} sections from {f}.")
                output = {
                    "name": f,
                    "title": title_text,
                    "abstract": abstract_text,
                    "metadata": {
                        "section": [],
                    },
                }

                last_section_title = ""

                for n in range(len(docs)):
                    current_section_title = docs[n].metadata["section_title"]
                    current_text = docs[n].metadata["text"]
                    current_section_number = docs[n].metadata["section_number"]

                    if current_section_number == "None":
                        # Match headings following pattern [A-Z](\.\d+)?
                        pattern = re.compile(
                            r"""
                                                ^\s*                                # Leading whitespace
                                                (?:Appendix\s+)?                    # Optional "Appendix"
                                                (                                   # group 1: numbering part
                                                    [A-Z](?:\.[0-9]+)*              # A, A.1, A.2.3...
                                                    |                               # or
                                                    \d+(?:\.\d+)*                   # 1, 1.1, 1.2.3...
                                                )
                                                (?:\s+(.*))?                        # group 2: title text (optional)
                                                $                                   # End of line
                                            """,
                            re.VERBOSE | re.IGNORECASE | re.MULTILINE,
                        )

                        match = pattern.match(current_section_title)
                        if match:
                            current_section_number = match.group(1)
                            current_section_title = (
                                match.group(2)
                                if match.group(2)
                                else "Appendix - " + str(match.group(1))
                            )
                            print(
                                "🔫 Detected appendix section:",
                                current_section_number,
                                current_section_title,
                            )
                            print("Last section title:", last_section_title)
                        else:
                            # inline heading (equations, tables, figures)
                            # Check if section list is empty
                            if output["metadata"]["section"]:
                                output["metadata"]["section"][-1]["text"] += (
                                    " " + current_section_title + " " + current_text
                                )
                            else:
                                # If list is empty, create a new section
                                output["metadata"]["section"].append(
                                    {
                                        "number": "None",
                                        "heading": current_section_title,
                                        "text": current_text,
                                    }
                                )
                            continue

                    # Same section, append text
                    if current_section_title in last_section_title:
                        output["metadata"]["section"][-1]["text"] += " " + current_text
                        continue

                    # Non-empty list + section number not None + encountering new section
                    if not output["metadata"]["section"] or (
                        current_section_number != "None"
                        and current_section_title not in last_section_title
                    ):

                        output["metadata"]["section"].append(
                            {
                                "number": current_section_number,
                                "heading": current_section_title,
                                "text": current_text,
                            }
                        )

                    last_section_title = current_section_title

                # Sort sections by number
                def sort_key(section):
                    number = section["number"]
                    if number == "None":
                        return (float("inf"), 0)  # Place "None" at the end

                    # Handle numeric numbering (1, 1.1, 1.2.1, etc.)
                    if re.match(r"^\d+(\.\d+)*$", number):
                        parts = [int(x) for x in number.split(".")]
                        return (0, parts)  # Numeric numbering has priority

                    # Handle alphabetic numbering (A, A.1, B.2, etc.)
                    if re.match(r"^[A-Z](\.\d+)*$", number):
                        first_char = ord(number[0]) - ord("A")
                        if "." in number:
                            parts = [int(x) for x in number[2:].split(".")]
                            return (
                                1,
                                [first_char] + parts,
                            )  # Alphabetic numbering is second
                        else:
                            return (1, [first_char])

                    return (2, number)  # Other formats go to the end

                output["metadata"]["section"].sort(key=sort_key)

                # Calculate relative path to maintain directory structure
                rel_path = os.path.relpath(f, paper_dir)
                output_path = os.path.join(parsed_pdfs_dir, rel_path).replace(
                    ".pdf", ".json"
                )

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Write JSON output after processing all documents
                with open(
                    output_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(output, file, indent=4, ensure_ascii=False)

                print(
                    "Parsed successfully🌈. Saving at ",
                    output_path,
                )
            except Exception as e:
                # Record failed papers
                error_info = {
                    "file_path": f,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                failed_papers.append(error_info)
                tqdm.write(f"❌ Failed to process {f}: {str(e)}")
                continue

        # After processing one paper_dir, write the failure log for that directory
        if parse_log_dir and failed_papers:
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                json.dump(
                    {
                        "paper_dir": paper_dir,
                        "total_failures": len(failed_papers),
                        "total_pdfs": len(pdf_files),
                        "failures": failed_papers,
                    },
                    log_file,
                    indent=4,
                    ensure_ascii=False,
                )
            print(
                f"\n📋 Parse log for {os.path.basename(os.path.dirname(paper_dir))} saved to: {log_file_path}"
            )
            print(f"Total failures: {len(failed_papers)} / {len(pdf_files)}")
        elif parse_log_dir:
            print(
                f"\n✅ All papers in {os.path.basename(os.path.dirname(paper_dir))} processed successfully!"
            )


if __name__ == "__main__":

    cfg_parse = load_configs(["configs/parse.yaml"])
        
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--grobid_port", type=int, default=8070, required=True)
    args = argparser.parse_args()

    parse_pdf_with_grobid(
        pdf_path=cfg_parse.raw_paper_pdfs,
        parse_log_dir=cfg_parse.parse_log,
        grobid_url=f"http://localhost:{args.grobid_port}",
    )
