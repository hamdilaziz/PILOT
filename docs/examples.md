## Examples

### 1. Generic model on a mixed document sample

**Command**
```bash
python run_pilot.py --config configs/pilot_generic.json --image input_samples/other1.png --task ocr_with_boxes
```

<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>Prediction</b></td>
  </tr>
  <tr>
    <td><img src="../input_samples/other1.png" width="420"></td>
    <td><img src="../outputs/other1_ocr_with_boxes_boxes.png" width="420"></td>
  </tr>
</table>

<details>
  <summary><b>Recognized text</b></summary>

```text
MAHP
MASSACHUSETTS
HEALTH PLANS
MAHP 2021 Virtual Conference
Thursday, November 18, 2021
9:00 am - 11:30 am
Confirmation of Sponsorship
Company/Organization:
Main Contact:
Names of Company Reps Attending:
Address:
State and Zip:
City:
Phone:
Email Address (main contact):
Sponsorship Level
Presenting $15,000
Platinum - $6,500
Gold - $4,500
Silver - $2,500
Payment Method
Total Due:
Visa:*
Mastercard:*
Amex:*
Please Invoice*
Check Enclosed:
Visa:*
Exp Date:
Card Number:
Total Amount to be Paid:
Name on credit card:
Billing Address (inc. zip):
Signature:
How did you hear about this event? (check all that apply)
MAHP e-mail announcing event
Another company's email
MAHP newsletter
Referred by colleague/friend
Searched the web
Other
Your signature below confirms your intent to sponsor and responsibility for monies associated with the selected sponsorship level.
Signature
Date
(Please print, sign and submit the original signed copy of this application along with payment.)
Thank you!
If you have any questions, please contact Ann Chamberlin Labelle at chamberlin@mahp.com or 617.338.2244 x 112
Please note: No refunds will be issued after September 16, 2021.
40 Court Street - Suite 550, Boston, MA 02108, 617.338.2244
```

</details>

Generated files:

- `outputs/other1_ocr_with_boxes_raw.txt`
- `outputs/other1_ocr_with_boxes.txt`
- `outputs/other1_ocr_with_boxes.json`
- `outputs/other1_ocr_with_boxes_boxes.png`

### 2. IAM handwriting OCR with boxes

**Command**
```bash
python run_pilot.py --config configs/pilot_iam.json --image input_samples/iam.png --task ocr_with_boxes
```

<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>Prediction</b></td>
  </tr>
  <tr>
    <td><img src="../input_samples/iam.png" width="420"></td>
    <td><img src="../outputs/iam_ocr_with_boxes_boxes.png" width="420"></td>
  </tr>
</table>

<details>
  <summary><b>Recognized text</b></summary>

```text
THE Fourth Gospel was almost certainly
written in Greek. A modern text of the
Gospel represents the work of generations of
scholars who have compared the many many manu-
scripts of John and worked out the version
which is most likely to have been the original
wording. It is not possible to establish any one
text with absolute precision.
```

</details>

Generated files:

- `outputs/iam_ocr_with_boxes_raw.txt`
- `outputs/iam_ocr_with_boxes.txt`
- `outputs/iam_ocr_with_boxes.json`
- `outputs/iam_ocr_with_boxes_boxes.png`

### 3. IAM spotting with `<find_it>`

**Command**
```bash
python run_pilot.py --config configs/pilot_iam_spotting.json --image input_samples/iam-spotting.png --task find_it --query "accidentally"
```

<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>Prediction</b></td>
  </tr>
  <tr>
    <td><img src="../input_samples/iam-spotting.png" width="420"></td>
    <td><img src="../outputs/iam-spotting_find_it_boxes.png" width="420"></td>
  </tr>
</table>

Generated files:

- `outputs/iam-spotting_find_it_raw.txt`
- `outputs/iam-spotting_find_it.json`
- `outputs/iam-spotting_find_it_boxes.png`

### 4. RIMES OCR with boxes

**Command**
```bash
python run_pilot.py --config configs/pilot_rimes.json --image input_samples/rimes.png --task ocr_with_boxes
```

<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>Prediction</b></td>
  </tr>
  <tr>
    <td><img src="../input_samples/rimes.png" width="420"></td>
    <td><img src="../outputs/rimes_ocr_with_boxes_boxes.png" width="420"></td>
  </tr>
</table>

<details>
  <summary><b>Recognized text</b></summary>

```text
NOVAIS Stépanie
16 RUE DE LA VALLEE
57800 ROSBRUCK
tel : 03770072994
Les 3 Suifs
40250 LAUREDE
Objet : Modification de commande
Ref : Lot de 3 paires de chaussettes (CH45-12)
n° Client : MZZQD86
Madame, Monsieur,
Ayant, en vain, essayer de contacter votre service
client je me permet de vous écrire afin d'effectuer
une modification de ma commande d'un lot de
3 paires de chaussettes (CH45-12). En effet, je
souhaiterai apporter une modification de la taille
et des couleurs. Je voudrai recevoir un lot
de 3 paires de chaussettes de couleur noir (re)ble et
de taille 39140 au lieu du lot de 3 paires de
chaussettes de couleur blanc/same/orange de taille 36/37.
Je vous prie de bien vouloir prendre en compte
ces modifications et d'agréer Madame, Monsieur mes
sincères salutations.
```

</details>

Generated files:

- `outputs/rimes_ocr_with_boxes_raw.txt`
- `outputs/rimes_ocr_with_boxes.txt`
- `outputs/rimes_ocr_with_boxes.json`
- `outputs/rimes_ocr_with_boxes_boxes.png`

### 5. SROIE OCR with boxes

**Command**
```bash
python run_pilot.py --config configs/pilot_sroie.json --image input_samples/sroie.jpg --task ocr_with_boxes
```

<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>Prediction</b></td>
  </tr>
  <tr>
    <td><img src="../input_samples/sroie.jpg" width="420"></td>
    <td><img src="../outputs/sroie_ocr_with_boxes_boxes.png" width="420"></td>
  </tr>
</table>

<details>
  <summary><b>Recognized text</b></summary>

```text
MR. D.I.Y. (KUCHAI) SDN BHD
CO-REG:750441-W
LOT 1851-A & 1851-B, JALAN KPB 6,
KAWASAN PERINDUSTRIAN BALAKONG,
43300 SERI KEMBANGAN, SELANGOR
(GST ID NO :000473792512)
(KEPONG)
-TAX INVOICE-
PLACEMAT 30*45CM
MB41/2 - 12/144
9067425
ITEM(S) : 1
TOTAL INCL. GST@6%
CASH
CHANGE
GST @6% INCLUDED IN TOTAL
QTY(S) : 2
RM 7.00
RM 10.00
RM 3.00
RM 0.40
11-01-16 14:00 SH01 B003
OPERATOR TRAINEE
EXCHANGE ARE ALLOWED WITHIN
3 DAY WITH RECEIPT.
STRICTLY NO CASH REFUND.
3.50
T1 R000375723
T1 R000375723
RM 0.40
EXCHANGE ARE ALLOWED WITHIN
3 DAY WITH RECEIPT.
STRICTLY NO CASH REFUND.
3.50
7.00
QTY(S) : 2
-TAX INVOICE-
```

</details>

Generated files:

- `outputs/sroie_ocr_with_boxes_raw.txt`
- `outputs/sroie_ocr_with_boxes.txt`
- `outputs/sroie_ocr_with_boxes.json`
- `outputs/sroie_ocr_with_boxes_boxes.png`

### 6. Region-conditioned OCR with `<ocr_on_box>`

**Command**
```bash
python run_pilot.py --config configs/pilot_generic.json --image input_samples/focus_en_bench.png --task ocr_on_box --box 50 836 737 851
```

<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>Selected region</b></td>
  </tr>
  <tr>
    <td><img src="../input_samples/focus_en_bench.png" width="420"></td>
    <td><img src="../outputs/focus_en_bench_ocr_on_box_input_box.png" width="420"></td>
  </tr>
</table>

<details>
  <summary><b>Recognized text</b></summary>

```text
prevais in art. He courman in asserton, award a la on a propriete portion of resonable atomers. He person or public public public.
```

</details>

Generated files:

- `outputs/focus_en_bench_ocr_on_box_raw.txt`
- `outputs/focus_en_bench_ocr_on_box.txt`
- `outputs/focus_en_bench_ocr_on_box.json`
- `outputs/focus_en_bench_ocr_on_box_input_box.png`

This task reads only the content inside the user-provided box.
