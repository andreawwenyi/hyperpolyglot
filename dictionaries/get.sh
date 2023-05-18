#!/bin/bash

for lang1 in en de es fr pt it
do
    for lang2 in en de es fr pt it
    do
	if [ $lang1 != $lang2 ]
	then
	    echo $lang1 $lang2
	    #wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$lang1-$lang2.txt
	fi
    done
done

for lang in af sq ar bn bs bg ca zh hr cs da nl et tl fi el he hi hu id ja ko lv lt mk ms no fa pl ro ru sk sl sv ta th tr uk vi
do
    echo $lang
    wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.txt
    wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$lang-en.txt
done
