import csv

with open('trackData2.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("|") for line in stripped if line)
    
    with open('trackData.csv', 'w',encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('trackID', 'AlbumId','ArtistId','genreId_1','genreId_2','genreId_3'))
        writer.writerows(lines)
