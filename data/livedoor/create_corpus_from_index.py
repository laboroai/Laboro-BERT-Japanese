#!/usr/bin/python

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sourcedir', help="livedoor corpus root directory")
    parser.add_argument('indexfile', help="index file {train,test,dev}_indexed.tsv")
    args = parser.parse_args()

    with open(args.indexfile) as f:
        f.readline() # throw away the header
        print("text\tlabel")
        for indexline in f:
            filename, category = indexline.rstrip().split()
            sourcefile = os.path.join(args.sourcedir, category, filename)
            if not os.path.exists(sourcefile):
                OSError("file {} not found".format(sourcefile))

            with open(sourcefile) as sf:
                lines = sf.readlines()[2:]
                text = ""
                for line in lines:
                    text += line.strip().replace("\t", " ")
                if '"' in text:
                    text = '"' + text.replace('"', '""') + '"'
                print("\t".join([text, category]))

