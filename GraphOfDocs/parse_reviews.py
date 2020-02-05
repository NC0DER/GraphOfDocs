import os
import sys
import string
import platform
from pathlib import Path
from utils import clear_screen

"""
Generator function to iterate over all occurences of substring in a string.
""" 
def find_all(string, sub):
    start = 0
    while True:
        start = string.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

"""
Function that extracts values, for a given type of xml tag, from an xml string.
If the tag is empty then None is returned.
""" 
def get_tag_value(string, type, start, end):
    # Construct tags from tag type.
    xml_start_tag = ''.join(('<', type, '>'))
    xml_end_tag = ''.join(('</', type, '>'))

    # Find the index of the starting tag.
    tag_start_idx = string.find(xml_start_tag, start, end)
    # Adjust the index to point after the starting tag.
    tag_start_idx = tag_start_idx + len(xml_start_tag)
    # Find the index of the ending tag.
    tag_end_idx = string.find(xml_end_tag, start, end)

    # Starting or ending tag not found, no value to be found.
    if tag_start_idx == -1 or tag_end_idx == -1:
        value = None
    # Both tags have been found but their distance is 0.
    # Which means, that they have no value between them.
    # E.g <tag></tag>
    elif (tag_end_idx - tag_start_idx) == 0:
        value = None
    else:
        # Return the value from tags by slicing the string.
        value = string[tag_start_idx:tag_end_idx]
    return value

"""
Function that read an xml-like-syntax file containing multiple amazon review xml tags,
which are processed to have their plaintext extracted, without any metadata.
Finally, each review is being written into its own file, in the output directory.
"""
def convert_xml_reviews_to_files(filepath):
    current_system = platform.system()
    with open(filepath, 'rt', encoding = 'utf-8-sig', errors = 'ignore') as file:
        # Read file contents and remove newline characters.
        text = file.read().replace('\n', ' ').replace('\r', '')
        # Remove non-printable characters from string.
        text = ''.join(filter(lambda x: x in string.printable, text))
        # Find the filename, and remove the .review extension.
        filename = os.path.basename(filepath)
        if filename.endswith('.review'):
            filename = filename[:-7]

        count = 1
        total_count = text.count('<review>')
        offset = len('<review>')
        # Iterate through text that has matching xml tags of <review></review>
        for start, end in zip(find_all(text, '<review>'), find_all(text, '</review>')):
            # Print the number of the currently processed review.
            print('Processing ' + str(count) + ' out of ' + str(total_count) + ' reviews...' )
            # Adjust the index to point after the starting tag.
            start = start + offset

            # Retrieve all content from these xml tags.
            product_name = get_tag_value(text, 'product_name', start, end)
            title = get_tag_value(text, 'title', start, end)
            review_text = get_tag_value(text, 'review_text', start, end)

            # Join all review information on a single string, then write it in a file.
            document_text = '\n'.join((product_name, title, review_text))

            # If the filename already exists, the file will be overwritten.
            # All files will be created in the default folder: output/
            # If the directory already exists, there will be raised no exception.
            new_filename = '_'.join((filename, str(count)))
            Path('output').mkdir(exist_ok = True)
            with open(''.join(('output/', new_filename)), 'w') as document:    
                document.write(document_text)
            count = count + 1
            # Clear the screen to output the update the progress counter.
            clear_screen(current_system)
    return
    
if __name__ == '__main__':
    if(len(sys.argv) > 1):
        # Filepath is expected to be the 2nd argument.
        convert_xml_reviews_to_files(sys.argv[1])
    else:
        print('Please input a file path, after parse_reviews.py.')
