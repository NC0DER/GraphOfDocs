# Graph-of-docs text representation

This repository hosts code for the papers:
* [On a novel representation of multiple textual documents in a single graph (KES-IDT 2020) - master_idt_2020 branch]()
* [An innovative graph-based approach to advance feature selection from multiple textual documents (AIAI 2020)]()

![image1](https://github.com/NC0DER/GraphOfDocs/blob/master/GraphOfDocs/images/GraphofDocs.jpg)

## Test Results
Run `jupyter notebook experiments.ipynb` 

## Dataset
Available in [this link](https://drive.google.com/drive/folders/121dlySvdaNSCoLOTJB2Cqt9cwf2nLoPq)

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
You can find the bibtex in [this link](), should you want to cite this paper.

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Nikos Kanakaris (nkanakaris@upnet.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
