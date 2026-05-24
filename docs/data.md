# Liiklusloenduse andmed

Liiklusloendusseadmete poolt kogutud liiklussageduste ja kiiruste andmed.

Siin andmestikus on ainult püsiloenduspunktide andmed.

Vaata ka:

- [Avaandmed - Loendusseadmete andmestik](https://avaandmed.eesti.ee/datasets/liiklusloendusseadmed)
- [Transpordiamet - Liiklussagedus](https://transpordiamet.ee/liiklussagedus)
- [Transpordiamet - Liiklussageduse statistika](https://transpordiamet.ee/liiklussageduse-statistika)

## Andmestikus olevad failid

- `ll_<aasta>.csv` - Liiklusloenduse andmed vastava aasta kohta
  - Praeguse aasta failis on andmed kuni jooksva kuu alguseni
- `ll_<aasta>.csv töödeldud` - Avaandmete poolt töödeldud versioon failist; API tarbimine võib töötada, aga ei soovita faili ennast alla laadida
- `ll_metadata.json` - Veerupõhised metaandmed csv failide kohta, mis peaks olema ka kättesaadav kui tarbida andmeid läbi API

## Andmete kuju

Igal real on ühe mõõteseadme ühe raja ühe tunni mõõtmistulemused:

- `id` - Mõõteseadme ID, viitab eraldi mõõteseadmete andmestikku
- `kanal` - Millist sõidurada mõõdeti
- `aeg` - Mõõtmise aeg, tunni täpsusega
- `1` - "Motorcycle" (mootorratas) sõidukite arv
- `2` - "Car / Light Van" (sõiduauto) sõidukite arv
- `3` - "Car/Lt. Van+Trailer" (sõiduauto + haagis) sõidukite arv
- `4` - "Heavy Van" (pakiauto) sõidukite arv
- `5` - "Light Goods" (väike veoauto) sõidukite arv
- `6` - "Rigid" (veoauto) sõidukite arv
- `7` - "Rigid + Trailer" (veoauto + haagis) sõidukite arv
- `8` - "Articulated HGV" (sadulrong) sõidukite arv
- `9` - "Minibus" (väikebuss) sõidukite arv
- `10` - "Bus / Coach" (buss) sõidukite arv
- Ülejäänud veerud (`<40Kph` kuni `=>130`) loevad kokku, mitu sõidukit sõitis vastava kiirusega

Hea teada:

- Sõiduki tüübi järgi ja sõiduki kiiruse järgi sõidukeid kokku lugedes peaks tulema sama arv
  - Nt. 2018 esimene rida: tüübi järgi on 39+7=46 ning kiiruse järgi on 1+8+18+16+3=46

## Andmete Excelisse importimine

Seda faili ei ole soovitatav topelt klõpsuga eestipärases Excelis avada: ta ei oska sellega midagi peale hakata (muu regiooni Excelis võib minna paremini).

Parima tulemuse jaoks:

- Ava Excelis tühi töövihik
- Vali Andmed - Too andmed - Failist - Teksti-/CSV-failist
- Impordi soovitud `ll_<aasta>.csv` fail
- Määra õiged parameetrid:
  - Faili päritolu: `65001: Unicode (UTF-8)`
  - Eraldaja: Koma
- Klõpsa "Andmeid transformeerima"
  - Klõpsa "Kasuta esimest rida päistena" kui "Esiletõstetud päised" ei ole juba etappides
  - Kontrolli üle, et andmetüübid on õiged (tõenäoliselt ebavajalik)
    - Andmetüübi muutmiseks klõpsa veeru nimest vasakule jäävat tüübi ikooni
    - id on tekst, aeg on ajavormis ning kõik muu on täisarvud
  - Klõpsa "Sule ja laadi"
- Kui "Päringud ja ühendused" kaardil ei ole näha ühtegi tõrget, siis läks laadimine loodetavasti edukalt

## Avaandmete lehe masintöötlusest

- Avaandmete leht nõuab pealkirja ja kirjeldust nii eesti kui inglise keeles, aga ikkagi automaatselt tõlgib eestikeelse teksti ingliskeelseks ja siis veel on tal jultumust öelda et "Metaandmete tõlkimiseks on kasutatud masintõlget ning nende kvaliteet võib seega olla kohati ebakorrektne"
- Failide järjekord ei ole sama kui üleslaadimise järjekord; ma ei tea miks ta neid sellises järjekorras näitab
- Ära lae alla töödeldud faili
  - Abivalmilt pannakse igale väärtusele jutumärgid ümber, ehk fail läheb suuremaks ja midagi ei muutu paremaks
  - Veerud järjestatakse ümber mingis järjekorras, sest see polnud ju tähtis kuidas need algselt olid, ega ju?
  - Kui laadida csv fail üles läbi veebilehe (mitte API), siis lähevad andmed veel rohkem katki; aga selle detailid jätaksin juba lugeja avastada

# Traffic census data

Data regarding traffic volumes and speeds gathered by traffic counters.

This dataset contains only data for permanent counting points.

See also:

- [Open data - Counting points](https://avaandmed.eesti.ee/datasets/liiklusloendusseadmed)
- [Transport Administration - Traffic volumes](https://transpordiamet.ee/en/roads-waterways-airspace/traffic-management/traffic-volumes)
- [Transport Administration - Traffic census results](https://transpordiamet.ee/liiklussageduse-statistika) (only in Estonian)

## Files in the dataset

- `ll_<year>.csv` - Traffic census data for a given year
  - Current year data is until start of current month
- `ll_<year>.csv processed` - Version of the file that has been processed by the open data website; consuming via API may work, but downloading this file is not advised
- `ll_metadata.json` - Column-based metadata about the csv files which should also be available via API

## Data structure

Every row contains data for one counter, one lane, one hour:

- `id` - Counter ID, refers to a separate traffic counting device dataset
- `kanal` - Which lane was measured
- `aeg` - Measurement time, hourly precision
- `1` - "Motorcycle" vehicle count
- `2` - "Car / Light Van" vehicle count
- `3` - "Car/Lt. Van+Trailer" vehicle count
- `4` - "Heavy Van" vehicle count
- `5` - "Light Goods" vehicle count
- `6` - "Rigid" vehicle count
- `7` - "Rigid + Trailer" vehicle count
- `8` - "Articulated HGV" vehicle count
- `9` - "Minibus" vehicle count
- `10` - "Bus / Coach" vehicle count
- The other columns (`<40Kph` kuni `=>130`) count how many vehicles drove by at a given speed

Other notes:

- Summing vehicle counts by vehicle type and speed should yield the same number
  - E.g. First row of 2018 data: by type 39+7=46 and by speed 1+8+18+16+3=46

## Importing data into Excel

This file is not recommended to be opened in Estonian regional Excel by way of double clicking on it (other flavors that respect comma as a delimiter may fare better).

For best results (keep in mind I do not have English language Excel and some labels may be different):

- Open an empty spreadsheet in Excel
- Click on Data - Get Data - From File - From Text/CSV
- Import the desired `ll_<year>.csv` file
- Pick the correct parameters:
  - File Origin: `65001: Unicode (UTF-8)`
  - Delimiter: Comma
- Click "Transform Data"
  - Click "Use First Row as Header" if such a step is not already in applied steps
  - Check that column types are correct (probably not necessary)
    - To change type, click the type icon to the left of the column name, pick a type, and click "Replace current" in the window that opens
    - id is string, aeg is time and everything else is integer
  - Click "Close & Load"
- If the card on the right does not show any issues, then hopefully the loading was successful

## About Estonian open data machine processing

- The open data page requires a title and description both in Estonian and English, but then proceeds to overwrite the English text with an automatic translation; it then has the gall to say "Metadata is machine translated and may thus be of poor quality"
- The ordering of files is not the same as the order they were uploaded; I have no idea what order this is
- Do not download a processed file
  - Ever-so-helpfully every value gets quotation marks around it, which only serves to inflate the file size and do nothing else
  - Columns are reordered in some god-only-knows order, because it's not like that was an important part of the file that you spent time designing, right?
  - If you upload a csv file through the website (not the API), then the data is even more broken; but the details of that shall be left to the reader to discover
