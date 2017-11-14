for i in {0..19}
do
    python ocr.py courier-train.png bc.train test-$i-0.png
done
