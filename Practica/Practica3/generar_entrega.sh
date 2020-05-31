#!/bin/bash
cd doc
make
make
cd ..
cp doc/demo.pdf .
mv demo.pdf Memoria_Practica3_Alejandro_Manzanares_Lemus.pdf
zip -r practica3_manzanares_lemus_alejandro.zip ./*.py ./Memoria_Practica3_Alejandro_Manzanares_Lemus.pdf
