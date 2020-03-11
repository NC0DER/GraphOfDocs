import sys
import string
import platform
from pathlib import Path
from utils import clear_screen
from parse_reviews import find_all, get_tag_value

"""
Function that read an sgml-like-syntax file containing multiple reuters news sgml tags,
which are processed to have their plaintext extracted, without any metadata.
Finally, each news story is being written into its own file, in the output directory.
"""
def convert_sgml_news_to_files(filepath):
    current_system = platform.system()
    with open(filepath, 'rt', encoding = 'utf-8-sig', errors = 'ignore') as file:
        # Read file contents and remove newline characters.
        text = file.read().replace('\n', ' ').replace('\r', '')
        # Remove non-printable characters from string.
        text = ''.join(filter(lambda x: x in string.printable, text))

        count = 1
        total_count = text.count('<REUTERS')
        offset = len('<REUTERS')
        # Iterate through text that has matching sgml tags of <REUTERS></REUTERS>
        for start, end in zip(find_all(text, '<REUTERS'), find_all(text, '</REUTERS>')):
            # Print the number of the currently processed news story.
            print(f'Processing {count} out of {total_count} news stories...')
            # Adjust the index to point after the starting tag.
            start = start + offset

            # Retrieve all content from these sgml tags.
            topics = get_tag_value(text, 'TOPICS', start, end)
            title = get_tag_value(text, 'TITLE', start, end)
            news_text = get_tag_value(text, 'BODY', start, end)
            # If the news story lacks these fields proceed to the next one.
            if topics is None or title is None or news_text is None:
                count = count + 1
                clear_screen(current_system)
                continue

            # Remove tags that encapsulate topic values, and separate them with periods.
            topics = topics.replace('<D>', '').replace('</D>', '.')
            # Remove last period separator from string.
            topics = topics[:-1]
            # Join all plaintext information on a single string, then write it in a file.
            document_text = '\n'.join((title, news_text))

            # Each news story will have a filename consisting of <topics>_<current_count>
            filename = '_'.join((topics, str(count)))
            # If the filename already exists, the file will be overwritten.
            # All files will be created in the default folder: output/
            # If the directory already exists, there will be raised no exception.
            Path('output').mkdir(exist_ok = True)
            with open(''.join(('output/', filename)), 'w') as document:    
                document.write(document_text)
            count = count + 1
            # Clear the screen to output the update the progress counter.
            clear_screen(current_system)
    return
    
if __name__ == '__main__':
    if(len(sys.argv) > 1):
        # Filepath is expected to be the 2nd argument.
        convert_sgml_news_to_files(sys.argv[1])
    else:
        print('Please input a file path, after parse_news.py.')
