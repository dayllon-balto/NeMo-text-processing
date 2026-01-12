#!/usr/bin/env python3
"""
Simple script to test inverse text normalization.

Usage:
    python test.py "text to format"
    python test.py "my address is nineteen one nine five valerio street reseda california"
    python test.py "send email to david dot ayllon at gmail dot com"
"""

import sys
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer


def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py \"text to format\"")
        print("\nExamples:")
        print("  python test.py \"one twenty three main street\"")
        print("  python test.py \"david dot ayllon at gmail dot com\"")
        print("  python test.py \"i live in california\"")
        sys.exit(1)

    # Join all arguments in case the text wasn't quoted
    text = " ".join(sys.argv[1:])

    # Initialize the normalizer (this may take a moment on first run)
    itn = InverseNormalizer(lang="en", cache_dir="itn_cache")

    # Normalize the text
    result = itn.normalize(text)

    print(f"Input:  {text}")
    print(f"Output: {result}")


if __name__ == "__main__":
    main()
