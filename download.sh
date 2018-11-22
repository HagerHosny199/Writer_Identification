#!/bin/bash


[ -s Data2 ]||mkdir Data2
wget -P Data2/  --user="$1" --password="$2"  -A "*.png"  -nH --cut-dirs=100 -r -nc -np http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/
[ -s Data2/robots.txt.tmp ]&&rm Data2/robots.txt.tmp