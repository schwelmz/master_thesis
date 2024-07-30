# Masterarbeit Moritz Schweller

## Requirements
The following python packages need to be installed:
- matplotlib
- numpy
- configparser

## Usage
- Parameter werden in parameter.ini geändert.
- settings.py enthält Funktionen um Parameter einzulesen und auszugeben.
- computation.py löst die PDE numerisch mit Hilfe von Finiten Differenzen und speichert die Daten im "out"-Ordner (muss ggf. noch manuell erstellt werden).
- Starte Rechnung mit:
 ```bash
python3 computation.py parameter.ini
```
- visualization.py enthält Funktionen zum plotten der Lösung und Erstellen von Animationen.
- main.py importiert gespeicherte Daten und plottet das Ergebnis.
