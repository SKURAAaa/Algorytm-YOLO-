import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# Wczytanie modelu
# ============================================================

def wczytaj_model(sciezka_modelu: str):

    try:

        model = YOLO(sciezka_modelu)

        print(f"[INFO] Wczytano model: {sciezka_modelu}")

        return model

    except Exception as e:

        print(f"[ERROR] Nie można wczytać modelu: {e}")

        sys.exit(1)


# ============================================================
# Wczytanie obrazu
# ============================================================

def wczytaj_obraz(sciezka_obrazu: str) -> np.ndarray:

    obraz = cv2.imread(sciezka_obrazu)

    if obraz is None:

        raise FileNotFoundError(
            f"Nie można wczytać obrazu: {sciezka_obrazu}"
        )

    return obraz


# ============================================================
# Wczytanie strumienia
# ============================================================

def wczytaj_strumien(zrodlo: str):

    if zrodlo == "camera":

        cap = cv2.VideoCapture(0)

    else:

        cap = cv2.VideoCapture(zrodlo)

    if not cap.isOpened():

        raise RuntimeError(
            f"Nie można otworzyć źródła: {zrodlo}"
        )

    return cap


# ============================================================
# Detekcja
# ============================================================

def wykonaj_detekcje(model, obraz: np.ndarray):

    wyniki = model(
        obraz,
        verbose=False
    )

    return wyniki


# ============================================================
# Odczyt wyników
# ============================================================

def odczytaj_wyniki(wyniki) -> List[Dict[str, Any]]:

    wynik = wyniki[0]

    detekcje = []

    boxes = wynik.boxes

    for box in boxes:

        xyxy = box.xyxy[0].cpu().numpy()

        conf = float(box.conf[0].cpu().numpy())

        cls = int(box.cls[0].cpu().numpy())

        detekcje.append({

            "box": xyxy,

            "class_id": cls,

            "confidence": conf

        })

    return detekcje


# ============================================================
# Filtrowanie confidence
# ============================================================

def filtruj_detekcje(
    detekcje: List[Dict[str, Any]],
    prog_confidence: float = 0.5
) -> List[Dict[str, Any]]:

    wynik = []

    for detekcja in detekcje:

        if detekcja["confidence"] >= prog_confidence:

            wynik.append(detekcja)

    return wynik


# ============================================================
# Nazwy klas
# ============================================================

def pobierz_nazwy_klas(model) -> Optional[Dict[int, str]]:

    return model.names


# ============================================================
# Rysowanie detekcji
# ============================================================

def rysuj_detekcje(
    obraz: np.ndarray,
    detekcje: List[Dict[str, Any]],
    nazwy_klas: Optional[Dict[int, str]] = None
) -> np.ndarray:

    wynik = obraz.copy()

    for detekcja in detekcje:

        x1, y1, x2, y2 = map(
            int,
            detekcja["box"]
        )

        class_id = detekcja["class_id"]

        confidence = detekcja["confidence"]

        if nazwy_klas:

            label = nazwy_klas[class_id]

        else:

            label = str(class_id)

        text = f"{label}: {confidence:.2f}"

        cv2.rectangle(
            wynik,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            wynik,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return wynik


# ============================================================
# Diagnostyka
# ============================================================

def dodaj_diagnostyke(
    obraz: np.ndarray,
    liczba_detekcji: int,
    prog_confidence: float
) -> np.ndarray:

    cv2.putText(
        obraz,
        f"Detekcje: {liczba_detekcji}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        obraz,
        f"Confidence: {prog_confidence}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    return obraz


# ============================================================
# Sprawdzenie datasetu
# ============================================================

def sprawdz_strukture_danych(sciezka_danych: str) -> bool:

    if not os.path.exists(sciezka_danych):

        print("[ERROR] Nie znaleziono data.yaml")

        return False

    return True


# ============================================================
# Informacje o anotacji
# ============================================================

def wypisz_informacje_o_anotacji():

    print("\n=== ANOTACJA YOLO ===")

    print("Narzędzia:")
    print("- LabelImg")
    print("- CVAT")
    print("- Makesense.ai")
    print("- Roboflow")

    print("\nFormat:")
    print("<class_id> <x_center> <y_center> <width> <height>")


# ============================================================
# Trening
# ============================================================

def dotrenuj_model(
    model,
    sciezka_danych: str,
    liczba_epok: int = 20,
    rozmiar_obrazu: int = 640
):

    if not sprawdz_strukture_danych(sciezka_danych):

        return

    model.train(

        data=sciezka_danych,

        epochs=liczba_epok,

        imgsz=rozmiar_obrazu,

        batch=8,

        name="bottle_detector"
    )

    print("\n[INFO] Trening zakończony")

    print(
        "runs/detect/bottle_detector/weights/best.pt"
    )


# ============================================================
# Model dotrenowany
# ============================================================

def wczytaj_model_dotrenowany(sciezka_modelu: str):

    return YOLO(sciezka_modelu)


# ============================================================
# Porównanie modeli
# ============================================================

def porownaj_modele(
    model_bazowy,
    model_dotrenowany,
    obraz: np.ndarray,
    prog_confidence: float
) -> Tuple[np.ndarray, np.ndarray]:

    # Bazowy
    wyniki_bazowe = wykonaj_detekcje(
        model_bazowy,
        obraz
    )

    detekcje_bazowe = filtruj_detekcje(
        odczytaj_wyniki(wyniki_bazowe),
        prog_confidence
    )

    obraz_bazowy = rysuj_detekcje(
        obraz,
        detekcje_bazowe,
        pobierz_nazwy_klas(model_bazowy)
    )

    # Dotrenowany
    wyniki_custom = wykonaj_detekcje(
        model_dotrenowany,
        obraz
    )

    detekcje_custom = filtruj_detekcje(
        odczytaj_wyniki(wyniki_custom),
        prog_confidence
    )

    obraz_custom = rysuj_detekcje(
        obraz,
        detekcje_custom,
        pobierz_nazwy_klas(model_dotrenowany)
    )

    return obraz_bazowy, obraz_custom


# ============================================================
# Obraz
# ============================================================

def przetwarzaj_obraz(
    model,
    sciezka_obrazu: str,
    prog_confidence: float
):

    obraz = wczytaj_obraz(
        sciezka_obrazu
    )

    wyniki = wykonaj_detekcje(
        model,
        obraz
    )

    detekcje = odczytaj_wyniki(
        wyniki
    )

    detekcje = filtruj_detekcje(
        detekcje,
        prog_confidence
    )

    obraz_wynikowy = rysuj_detekcje(
        obraz,
        detekcje,
        pobierz_nazwy_klas(model)
    )

    obraz_wynikowy = dodaj_diagnostyke(
        obraz_wynikowy,
        len(detekcje),
        prog_confidence
    )

    cv2.imshow(
        "Wynik detekcji - obraz",
        obraz_wynikowy
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# Wideo / kamera
# ============================================================

def przetwarzaj_wideo(
    model,
    zrodlo: str,
    prog_confidence: float
):

    cap = wczytaj_strumien(zrodlo)

    while True:

        poprawnie, klatka = cap.read()

        if not poprawnie:

            break

        wyniki = wykonaj_detekcje(
            model,
            klatka
        )

        detekcje = odczytaj_wyniki(
            wyniki
        )

        detekcje = filtruj_detekcje(
            detekcje,
            prog_confidence
        )

        klatka_wynikowa = rysuj_detekcje(
            klatka,
            detekcje,
            pobierz_nazwy_klas(model)
        )

        klatka_wynikowa = dodaj_diagnostyke(
            klatka_wynikowa,
            len(detekcje),
            prog_confidence
        )

        cv2.imshow(
            "Wynik detekcji - wideo",
            klatka_wynikowa
        )

        klawisz = cv2.waitKey(1) & 0xFF

        if klawisz in (ord("q"), 27):

            break

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# Porównanie
# ============================================================

def uruchom_porownanie(
    model_bazowy,
    model_dotrenowany,
    sciezka_obrazu: str,
    prog_confidence: float
):

    obraz = wczytaj_obraz(
        sciezka_obrazu
    )

    obraz_bazowy, obraz_dotrenowany = porownaj_modele(

        model_bazowy,

        model_dotrenowany,

        obraz,

        prog_confidence
    )

    cv2.imshow(
        "Model bazowy",
        obraz_bazowy
    )

    cv2.imshow(
        "Model dotrenowany",
        obraz_dotrenowany
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="LAB6 - YOLO"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ścieżka do modelu"
    )

    parser.add_argument(
        "--image",
        help="Ścieżka do obrazu"
    )

    parser.add_argument(
        "--video",
        help="Ścieżka do filmu"
    )

    parser.add_argument(
        "--camera",
        action="store_true",
        help="Użyj kamery"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Uruchom trening"
    )

    parser.add_argument(
        "--train-data",
        help="Ścieżka do data.yaml"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--trained-model",
        help="Ścieżka do modelu dotrenowanego"
    )

    parser.add_argument(
        "--compare",
        action="store_true"
    )

    parser.add_argument(
        "--show-annotation-help",
        action="store_true"
    )

    args = parser.parse_args()

    # Model bazowy
    model = wczytaj_model(
        args.model
    )

    # anotacje
    if args.show_annotation_help:

        wypisz_informacje_o_anotacji()

    # Trening
    if args.train:

        if not args.train_data:

            print("[ERROR] Podaj --train-data")

            sys.exit(1)

        dotrenuj_model(

            model,

            args.train_data,

            args.epochs,

            args.imgsz
        )

        return

    # Porównanie
    if args.compare:

        if not args.trained_model:

            print("[ERROR] Podaj --trained-model")

            sys.exit(1)

        if not args.image:

            print("[ERROR] Podaj --image")

            sys.exit(1)

        model_dotrenowany = wczytaj_model_dotrenowany(
            args.trained_model
        )

        uruchom_porownanie(

            model,

            model_dotrenowany,

            args.image,

            args.confidence
        )

        return

    # Obraz
    if args.image:

        przetwarzaj_obraz(

            model,

            args.image,

            args.confidence
        )

        return

    # Wideo
    if args.video:

        przetwarzaj_wideo(

            model,

            args.video,

            args.confidence
        )

        return

    # Kamera
    if args.camera:

        przetwarzaj_wideo(

            model,

            "camera",

            args.confidence
        )

        return

    print("[ERROR] Nie podano trybu działania")


if __name__ == "__main__":

    raise SystemExit(main())