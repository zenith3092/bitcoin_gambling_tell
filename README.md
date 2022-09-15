# bitcoin_gambing_predict
## Introduction
This is a simple package to get the result of each algorithm.

## Environment
Python Versrion : 3.9+

Use this command on terminal to install necessary modules : 
```
pip3 install -r requirements.txt
```

## Execute program file
### Step 1 :
Use this command on terminal to execute program file:
```
python3 main.py
```
### Step 2 :
After you execute this python file, you will see the message:
```
Please insert the algorithms you want to use:
```
This time you can insert keyword of algorithm. 

If you want to execute more than two algorithms, please segment them with commas.

And if you want to execute all the algorithm, please insert "all".

### Step 3 :
Then, you will also see the message:
```
Do you want to predict? (y / other value)
```
If you want to use these algorithms to predict, please insert "y".

If not, pleasw insert any other value except "y".

### Step 4 :
You will see some images and csv files created.

## How do we create features?
### Step 1 : Prepare classified tx ID data
classify_tx_data.py

=> get_tx_data.py

=> get_tx_info.py

### Step 2 : Prepare classified address data
get_address_classfied.py

=> get_features

### Step 3 : merge sheet2 and sheet3 (tag the address)
merge_data.py

## Reference
### support_algorithm.yaml
You can look up the keyword of algorithm in this file.

Beside, if you want to stop some algorithms from executing, comment out them in this file.

### column explanation.txt
You can look up the column explanation in this file.


