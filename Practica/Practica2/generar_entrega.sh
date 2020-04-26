#!/bin/bash
cd doc
make
make
cd ..
cp doc/demo.pdf .
mv demo.pdf Memoria_Practica2_Alejandro_Manzanares_Lemus.pdf
cp py/* .
zip -r practica2_manzanares_lemus_alejandro.zip ./*.py ./Memoria_Practica2_Alejandro_Manzanares_Lemus.pdf
rm *.py
