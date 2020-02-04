import sys
import json
import string
import platform
from pathlib import Path
from utils import clear_screen

"""
Function that reads an json file containing multiple jira issues,
which are processed to have their plaintext extracted, without any metadata.
Finally, each issue is being written into its own file, in the output directory.
"""
def convert_json_issues_to_files(filepath):
    current_system = platform.system()
    # List that contains the chosen assignees
    assignees = [
        'aantonenko', 'andrus', 'atkach', 'wesmckinn', 'julianhyde',
        'andy.seaborne', 'purplecabbage', 'ababiichuk', 'jbellis',
        'batik-dev@xmlgraphics.apache.org', 'djohnson', 'ancosen',
        'elserj', 'bowserj', 'onechiporenko']

    with open(filepath, 'rt', encoding = 'utf-8-sig', errors = 'ignore') as dataset:
        # Load the json object in memory as a list of dictionaries.
        issues = json.load(dataset)['issues']
        count = 1
        skip = 0 
        total_count = len(issues)
        # Iterate all issues.
        for issue in issues:
            # Retrieve all important fields from the dictionary.
            # Print the number of the currently processed issue.
            print('Processing ' + str(count) + ' out of ' + str(total_count) + ' issues...' )
            issue_key = issue['key']
            issue_summary = ('' if issue['fields']['summary'] is None else issue['fields']['summary'])
            issue_description = ('' if issue['fields']['description'] is None else issue['fields']['description'])
            assignee_key = issue['fields']['assignee']['key']
            assignee_key = assignee_key.translate({ord(c): '' for c in '\'\"/*:?<>|_'})

            # Choose the top 15 assignees of the dataset.
            if assignee_key not in assignees:
                count = count + 1
                clear_screen(current_system())
                continue

            # Each issue will become a file.
            # The filename is derived from the following convention.
            # <assignee_key>_<issue_key>
            file_name = '_'.join((assignee_key, issue_key))
            document_text = '\n'.join((issue_summary, issue_description))
            document_text = ''.join(filter(lambda x: x in string.printable, document_text))

            # Skip issues with empty text.
            if document_text == '':
                skip = skip + 1
                continue

            # If the filename already exists, the file will be overwritten.
            # All files will be created in the default folder: output/
            # If the directory already exists, there will be raised no exception.
            Path('output').mkdir(exist_ok = True)
            with open(''.join(('output/', file_name)), 'w') as document:    
                document.write(document_text)
            count = count + 1
            # Clear the screen to output the update the progress counter.
            clear_screen(current_system)
        # Print Completed and skip items if any.
        print('Loaded ' + str(count - skip) + ' issues, skipped ' + str(skip) + ' empty items.')
    return

if __name__ == "__main__": convert_json_issues_to_files(sys.argv[1]) # Filepath is expected to be the 2nd argument.
