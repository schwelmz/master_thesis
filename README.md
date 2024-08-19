# Masterarbeit Moritz Schweller

## Requirements
The following python packages need to be installed:
- matplotlib
- numpy
- configparser

## File Structure
- Parameter werden in parameter.ini geändert.
- settings.py enthält Funktionen um Parameter einzulesen und auszugeben.
- computation.py löst die PDE numerisch mit Hilfe von Finiten Differenzen und speichert die Daten im "out"-Ordner (sollte automatisch erstellt werden).
- visualization.py enthält Funktionen zum plotten der Lösung und Erstellen von Animationen.
- main.py importiert gespeicherte Daten und plottet das Ergebnis.

## Usage
- Starte einfache Rechnung (keine plots, keine Zwischenspeicherungen, nur Endresultat als npy-file) mit:
```bash
python3 computation.py -p parameter.ini
```
- spezifiziere initial condition ("4dots", "randmon-dots" oder "white-noise"), default ist white-noise
```bash
python3 computation.py -p parameter.ini -init white-noise
```
- spezifiziere output Ordner, in dem die Ergebnisse gespeichert werden (default = simulation_results). Ergebnisse werden dann in "out/<dir_name>/" gespeichert.
 ```bash
python3 computation.py -p parameter.ini -init white-noise -o my_output_folder
```
- aktiviere Zwischenspeichern und automatisches Erzeugen von 250 plots (zum Erstellen von Videos):
```bash
python3 computation.py -p parameter.ini -init white-noise -o my_output_folder --videomode
```
- Setze Rechnung von alten Ergebnissen (input files) fort:
```bash
python3 computation.py -p parameter.ini -init white-noise -o my_output_folder -i path/to/Nodal_input.npy path/to/Lefty_input.npy <starting time [min]> --videomode
```
- Wähle anderes Modell (Nodal-Lefty(NL) oder GiererMeinhardt(GM)). Default ist NL.
 ```bash
python3 computation.py -p parameter.ini -init white-noise -o my_output_folder -m GM
```
