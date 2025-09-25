import requests
from datetime import datetime

def format_tanggal(tanggal_str: str) -> str:
    bulan_id = {
        "01": "Januari", "02": "Februari", "03": "Maret",
        "04": "April", "05": "Mei", "06": "Juni",
        "07": "Juli", "08": "Agustus", "09": "September",
        "10": "Oktober", "11": "November", "12": "Desember"
    }
    tgl = datetime.strptime(tanggal_str, "%Y-%m-%d")
    return f"{tgl.day} {bulan_id[tanggal_str[5:7]]} {tgl.year}"

# Endpoint API
url = "https://booking.forrizhotels.com/api/v2/offers/room"

# Header
headers = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJndWVzdCI6dHJ1ZSwiaWF0IjoxNzQ0MTAzMjk1LCJleHAiOjQ4OTc3MDMyOTUsImp0aSI6IjlveXZCemktdWRqZ3lSeDBOSy1KM1lSWnhTZkNuZ1hUX2dsLV94cExFM0E9In0.h0rWAisFpD6LmZzgh7l8h0ABG_fi_8wxZaULHFDu04U",
}

payload = {
    "checkin": "2025-09-26",
    "checkout": "2025-09-27",
    "id": "FHYH",
}

# Kirim POST request tanpa payload
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    rooms = data.get("room", [])

    checkin_text = format_tanggal(payload['checkin'])
    checkout_text = format_tanggal(payload['checkout'])
    print(f"Ketersediaan Kamar untuk tanggal : {checkin_text}\n")

    for idx, room in enumerate(rooms, start=1):
        room_name = room.get("name")
        available_room = room.get("available_room")
        bed_type = room.get("bed_type")

        print(f"{idx}. Tipe Kamar : {room_name}")
        print(f"Jumlah Tersedia : {available_room}")
        print(f"Jenis Tempat Tidur : {bed_type}")

        offers = room.get("offers", [])
        for offer in offers:
            offer_name = offer.get("name")
            offer_price = offer.get("price")
            print(f"Penawaran: {offer_name}, Harga: {offer_price}")

        print()
        
else:
    print("Request failed:", response.status_code, response.text)
    result_json = None