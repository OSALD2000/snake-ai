# Snake-KI

[English](README.md)

Ein kleines, reproduzierbares Snake-Experiment mit einem neuronalen Q-Learning-
Agenten. Es ist eine saubere Weiterentwicklung meines älteren Projekts `snake_ki`.
Spiel, Netz und Tests wurden für dieses Repository neu umgesetzt; die alte
X11-Steuerung, das kompilierte Spiel und die Modelldateien sind nicht enthalten.

## Starten

Benötigt wird nur Python 3.10 oder neuer. API-Schlüssel, Bildschirmserver und
zusätzliche Pakete sind nicht erforderlich.

```sh
python3 -m snake_ai.train --episodes 3000 --seed 0
python3 -m unittest discover -s tests -v
```

Das Training speichert `model.json` lokal; Git ignoriert die Datei. Mit
`--model PFAD`, `--episodes N` und `--eval-games N` lassen sich Speicherort,
Trainingslänge und Anzahl der Auswertungsspiele ändern.

## Funktionsweise

- Ein 8×8-Spielfeld liefert elf Zustandswerte: Gefahr geradeaus/rechts/links,
  Bewegungsrichtung und Richtung zum Futter.
- Der Agent wählt geradeaus, rechts oder links. Ein Netz mit einer ReLU-Schicht
  aus 16 Neuronen schätzt den Wert jeder Aktion.
- Online-Q-Learning aktualisiert das Netz nach jedem Zug. Die Zufallsrate sinkt
  während des Trainings von 100 % auf 5 %.
- Die Auswertung nutzt 100 neue Spielstarts und vergleicht den trainierten
  Agenten mit Zufallszügen bei jeweils höchstens 200 Zügen.

Hier wurden nach 3.000 Trainingsspielen folgende Werte gemessen:

| Trainings-Seed | Trainierter Agent | Zufallszüge |
| --- | ---: | ---: |
| 0 | 15,28 Äpfel/Spiel | 0,17 Äpfel/Spiel |
| 42 | 16,23 Äpfel/Spiel | 0,20 Äpfel/Spiel |

Das sind beobachtete Ergebnisse, keine Garantie für jeden Seed und kein allgemeiner
Snake-Benchmark. Die Belohnung berücksichtigt auch eine kleine Annäherung an das
Futter. Das Netz lernt online, ohne Experience Replay oder separates Zielnetz.
Nach 200 Zügen endet die Auswertung auch dann, wenn die Schlange noch lebt.
Das gespeicherte JSON-Modell ist nicht mit dem alten TensorFlow-Modell kompatibel.
