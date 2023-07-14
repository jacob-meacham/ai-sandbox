import argparse

import csv
from markdownify import markdownify


def clean_content(content):
    content = content.replace(u'\xa0', ' ')
    content = content.replace('[Back to top](#Top)\n', '')
    while len(content) and content[0].isspace():
        content = content[1:]

    return content


def main(infile, outfile):
    reader = csv.DictReader(infile)
    out = []
    for r in reader:
        content = clean_content(markdownify(r['ARTICLE_BODY__C']))

        if content:
            out.append({
                'id': r['URLNAME'],
                'title': r['TITLE'],
                'content': content,
                'product': r['PRODUCT__C']
            })

    writer = csv.DictWriter(outfile, out[0].keys())
    writer.writeheader()
    writer.writerows(out)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('r', encoding='utf-8-sig'))
    parser.add_argument('outfile', type=argparse.FileType('w'))
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()
    main(options.infile, options.outfile)
