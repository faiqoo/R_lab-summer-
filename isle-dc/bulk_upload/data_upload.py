import pandas as pd
import os
import xml.etree.ElementTree as ET
import dateparser
import requests
import warnings
import unicodedata

# Ignore all warnings
warnings.filterwarnings("ignore")

jwt_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpYXQiOjE3ODM2ODY1NzUsImV4cCI6MTc4MzY5Mzc3NSwiZHJ1cGFsIjp7InVpZCI6IjEifSwid2ViaWQiOiIxIiwiaXNzIjoiaHR0cHM6Ly9pc2xhbmRvcmEuZGV2Iiwic3ViIjoiYWRtaW4iLCJyb2xlcyI6WyJhdXRoZW50aWNhdGVkIiwiYWRtaW5pc3RyYXRvciIsImZlZG9yYWFkbWluIl0sImF1ZCI6WyJpc2xhbmRvcmEiXX0.hpIJfUoes-1M1IPOorbvvvq3Y_FsQwbor4Z0fS49luqkUpy7lX51mxmXdf01Wmsj2o5vQAR-_lzbAYCzrm0Ucc4I3KQuT9NFtCcVvN8Hhlugkl_0_cZ7GLWWNBFUJa4TrjSgCiQGhUi4fj22FXVYsqzdOu5WHnDKYRabaCuQZdla4cUwsKnEsgO-MXu6OIuDpb0D2NToSSQS3t9Gk0YmCctdFk7eZCellF89yINu4cK-aXkQ7-SZ2rjFK3WbWNsxQ7efUK9XT3DnNhnsoLADDfWpWXEQLUms45SfDJf_fK1DNZA-EHRumJGLFrNcMqSEc-PFmNuluzaWnnvrj225Bw"  # https://islandora.dev/jwt/token only the token value is required, not the full JSON

tei_directory = "./tei"
pdf_directory = "./pdfs"

# For creating nodes
NODE_ENDPOINT = "https://islandora.dev/de/node?_format=json"  # Set to actual URL
DIGITAL_DOCUMENT_ID = 904  # Set the correct value for your instance. (Check Structure/Taxonomy Terms/Islandora Models, Digital Document)
# For creating media
MEDIA_ENDPOINT = "https://islandora.dev/de/node/{node_id}/media/{media_type}/{media_use}"
MEDIA_USE_ORIGINAL_FILE_ID = 894  # Set the correct value for your instance. (Check Structure/Taxonomy Terms/Islandora Media Use, Original File)
MEDIA_USE_TRANSKRIPT = 898  # Set the correct value for your instance. (Check Structure/Taxonomy Terms/Islandora Media Use, Transcript)
PDF_MEDIA_TYPE = 'document' # Structure/Media, Document
TEI_MEDIA_TYPE = 'tei_xml' # Structure/Media, Document


VERIFY = False

def create_node(data):
    headers = {
        "Content-Type": "application/json",  # This sets the request content type
        "Authorization": f"Bearer {jwt_token}"
    }
    resp = requests.post(NODE_ENDPOINT, headers=headers, json=data, verify=VERIFY)  # , auth=HTTPBasicAuth(user, password)
    # print(resp.status_code)
    return resp

def attach_media(node_resp, file_path, media_type, media_use, headers):
    node_id = node_resp['nid'][0]['value']
    # print(f"Node-ID: {node_id}")
    # print(file_path)
    # The URL is built using the pattern: "http://localhost:8000/node/{node id}/media/{media type}/{media use}"
    target_url = MEDIA_ENDPOINT.format(node_id=node_id, media_type=media_type, media_use=media_use)
    # print(target_url)
    with open(file_path, "rb") as f:
        resp = requests.put(target_url, headers=headers, data=f, verify=VERIFY)
    return resp

with open("title_prefixes.txt", "r") as f:
    titel_anfang_liste = [l.strip() + " " for l in f.readlines()]

FIELD_MAPPING = {
            "title": "title",
            'field_fundort': "document_url",
            'field_beruf': "title",  # Beruf wird mit entsprechender Funktion aus dem Titel extrahiert
            'field_erlassdatum': "date",
            'field_erscheinungsjahr': "year",
            'field_erscheinungsort': "Erscheinungsort",
            'field_herausgeber': "Herausgeber",
            'field_seitenzahl': "Anzahl Seiten",
            'field_verlag': "Bundesanzeiger",
        }

def transform_beruf(beruf_string):
    for titel_anfang in titel_anfang_liste:
        if titel_anfang in beruf_string:
            beruf = beruf_string.replace(titel_anfang, "").split("(")[0].strip().split("/")[0].strip()
            return beruf
    return "-"

def safe_filename(filename):
    # Normalize unicode
    filename = unicodedata.normalize("NFKD", filename)

    # Remove problematic chars (umlauts → ascii)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    return filename

ns = {
    "tei": "http://www.tei-c.org/ns/1.0"
    }

records = []

for fi, filename in enumerate(os.listdir(tei_directory)):

    if not filename.endswith(".xml"):
        continue

    filepath = os.path.join(tei_directory, filename)

    tree = ET.parse(filepath)
    root = tree.getroot()

    data = {
        'field_verlag': [{"value": "Bundesanzeiger"}],
        'field_fundort': [{"value": 'https://offenegesetze.de/veroeffentlichung/bgbl1/'}],
        "type": [{"target_id": "aus_fortbildungsordnung"}],
        "field_model": [{"target_id": DIGITAL_DOCUMENT_ID, "target_type":"taxonomy_term"}]
        }

    # Year
    year = os.path.splitext(filename)[0].split("_")[-1]
    if year is not None and year != "":
        data["field_erscheinungsjahr"] = [{"value": year}]

    # Title
    title = root.find(".//tei:title", ns)
    if title is not None and title.text is not None:
        title_text = title.text.strip().replace("\n", " ")
        data["title"] = [{"value": title_text}]
        data["field_beruf"] = [{"value": transform_beruf(title_text)}]

    # Date
    date = root.find(".//tei:publicationStmt/tei:date", ns)
    if date is not None and date.text is not None:
        date_text = date.text.strip()
        try:
            parsed_date = dateparser.parse(date_text)
            data["field_erlassdatum"] = [{"value": parsed_date.date().strftime("%Y-%m-%d")}]
        except Exception:
            data["field_erlassdatum"] = None

    # Erscheinungsort
    pub_place = root.find(".//tei:publicationStmt/tei:pubPlace", ns)
    if pub_place is not None and pub_place.text is not None:
        data["field_erscheinungsort"] = [{"value": pub_place.text.strip()}]

    # Herausgeber
    org = root.find(".//tei:author/tei:orgName", ns)
    if org is not None and org.text is not None:
        data["field_herausgeber"] = [{"value": org.text.strip()}]

    # Seitenzahl (count pb tags)
    pb_tags = root.findall(".//tei:pb", ns)
    data["field_seitenzahl"] = [{"value": len(pb_tags)}]

    # print(data)
    records.append(data)
    print("------------------------------------------------------------")
    
    try:
        # Create node
        resp = create_node(data).json()
        print(resp)
        # upload pdf
        pdf_path = os.path.join(pdf_directory, os.path.splitext(filename)[0] + ".pdf")
        headers = {
            "Content-type": "application/pdf",
            "Content-Location": f"fedora://documents/{safe_filename(os.path.basename(pdf_path))}",
            "Authorization": f"Bearer {jwt_token}"
        }
        # print(pdf_path)
        # print(os.path.exists(pdf_path))
        resp_pdf = attach_media(resp, pdf_path, media_type=PDF_MEDIA_TYPE, media_use=MEDIA_USE_ORIGINAL_FILE_ID, headers=headers)
        print(resp_pdf.text)
        # # upload tei
        # tei_path = os.path.join(tei_directory, os.path.splitext(filename)[0] + ".xml")
        # headers = {
        #     "Content-type": "application/xml",
        #     "Content-Location": f"webdav://{safe_filename(os.path.basename(tei_path))}",
        #     "Authorization": f"Bearer {jwt_token}"
        # }
        #print(tei_path)
        #print(os.path.exists(tei_path))
        # resp_tei = attach_media(resp, tei_path, media_type=TEI_MEDIA_TYPE, media_use=MEDIA_USE_TRANSKRIPT, headers=headers)
        # print(resp_tei.text)
    except Exception as e:
        print(e)
        print(fi)
        print(f"Failed for index {data}")

result_df = pd.DataFrame(records)
