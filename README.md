# Graph-of-docs Text Representation

This repository hosts code for the papers:
* [On a novel representation of multiple textual documents in a single graph (KES-IDT 2020)](https://link.springer.com/chapter/10.1007%2F978-981-15-5925-9_9) - [Download](https://github.com/NC0DER/GraphOfDocs/releases/tag/KES-IDT-2020)
* [An innovative graph-based approach to advance feature selection from multiple textual documents (AIAI 2020)](https://link.springer.com/chapter/10.1007%2F978-3-030-49161-1_9) - [Download](https://github.com/NC0DER/GraphOfDocs/archive/master.zip)

![image1](https://github.com/NC0DER/GraphOfDocs/blob/master/GraphOfDocs/images/feature_selection.jpg)

## Datasets
Available in [this link](https://github.com/imis-lab/aiai-2020-datasets)

## Test Results
Edit `GraphOfdocs/config_experiments.py` to setup the experiments and run `experiments.py`.

## Installation
**Prequisites:**
* `Windows 10` 64-bit / Debian based `Linux` 64-bit.  
* `Python 3` (min. version 3.6), `pip3` (& `py` launcher Windows-only).  
* Working `Neo4j` Database (min. version 3.5.12).  

### Windows 10
Download the project from the green button above, unzip it,  
and then open a cmd terminal to this folder and type `pip3 install -r requirements.txt`.  
This command will install the neccessary `Python` libraries\* to run the project.  

### Debian Based Linux
We ran the following commands to update `Python`, `git`,  
clone the project to a local folder and install the necessary `Python` libraries\*.
```bash
sudo apt install python3.6
sudo apt install git-all
git clone https://github.com/NC0DER/GraphOfDocs
cd GraphOfDocs
pip3 install -r requirements.txt
```
*\* Optionally you could create a virtual environment first,*  
*\* to isolate the libraries from your python user install.*  
*\* However the setup script doesn't downgrade existing libraries,*  
*\* so there's zero risk in affecting your local user install.*  

## Database Setup (Windows / Linux)
Create a new database from the `Neo4j` desktop app using 3.5.12 as the min. version.  
Update your memory settings to match the following values,  
and install the following extra plugins as depicted in the image.
![image2](https://github.com/NC0DER/GraphOfDocs/blob/master/GraphOfDocs/images/settings.jpg)
*Hint: if you use a dedicated server that only runs `Neo4j`, you could increase these values, 
accordingly as specified in the comments of these parameters.*

Run the `GraphOfDocs.py` script which will create thousands of nodes, 
and millions of relationships in the database.  
Once it's done, the database is initialized and ready for use. 

## Running the app
You could use the `Neo4j Browser` to run your queries, 
or for large queries you could use the custom visualization tool  
`visualize.html` which is located in the `GraphOfDocs` Subdirectory.

## Citation
On a novel representation of multiple textual documents in a single graph (KES-IDT 2020) paper:
```
Giarelis N., Kanakaris N., Karacapilidis N. (2020) On a Novel Representation of Multiple Textual Documents in a Single Graph. In: Czarnowski I., Howlett R., Jain L. (eds) Intelligent Decision Technologies. IDT 2020. Smart Innovation, Systems and Technologies, vol 193. Springer, Singapore
```

```
@InProceedings{10.1007/978-981-15-5925-9_9,
author="Giarelis, Nikolaos
and Kanakaris, Nikos
and Karacapilidis, Nikos",
editor="Czarnowski, Ireneusz
and Howlett, Robert J.
and Jain, Lakhmi C.",
title="On a Novel Representation of Multiple Textual Documents in a Single Graph",
booktitle="Intelligent Decision Technologies",
year="2020",
publisher="Springer Singapore",
address="Singapore",
pages="105--115",
abstract="This paper introduces a novel approach to represent multiple documents as a single graph, namely, the graph-of-docs model, together with an associated novel algorithm for text categorization. The proposed approach enables the investigation of the importance of a term into a whole corpus of documents and supports the inclusion of relationship edges between documents, thus enabling the calculation of important metrics as far as documents are concerned. Compared to well-tried existing solutions, our initial experimentations demonstrate a significant improvement of the accuracy of the text categorization process. For the experimentations reported in this paper, we used a well-known dataset containing about 19,000 documents organized in various subjects.",
isbn="978-981-15-5925-9"
}
```

An innovative graph-based approach to advance feature selection from multiple textual documents (AIAI 2020) paper:
```
Giarelis N., Kanakaris N., Karacapilidis N. (2020) An Innovative Graph-Based Approach to Advance Feature Selection from Multiple Textual Documents. In: Maglogiannis I., Iliadis L., Pimenidis E. (eds) Artificial Intelligence Applications and Innovations. AIAI 2020. IFIP Advances in Information and Communication Technology, vol 583. Springer, Cham
```

```
@InProceedings{10.1007/978-3-030-49161-1_9,
author="Giarelis, Nikolaos
and Kanakaris, Nikos
and Karacapilidis, Nikos",
editor="Maglogiannis, Ilias
and Iliadis, Lazaros
and Pimenidis, Elias",
title="An Innovative Graph-Based Approach to Advance Feature Selection from Multiple Textual Documents",
booktitle="Artificial Intelligence Applications and Innovations",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="96--106",
abstract="This paper introduces a novel graph-based approach to select features from multiple textual documents. The proposed solution enables the investigation of the importance of a term into a whole corpus of documents by utilizing contemporary graph theory methods, such as community detection algorithms and node centrality measures. Compared to well-tried existing solutions, evaluation results show that the proposed approach increases the accuracy of most text classifiers employed and decreases the number of features required to achieve `state-of-the-art' accuracy. Well-known datasets used for the experimentations reported in this paper include 20Newsgroups, LingSpam, Amazon Reviews and Reuters.",
isbn="978-3-030-49161-1"
}
```

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Nikos Kanakaris (nkanakaris@upnet.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
