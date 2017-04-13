import csv
with open("trackData413.csv","r") as source:
    #rdr= [x.decode('utf8').strip() for x in source.readlines()]
    rdr= csv.reader( source )
    with open("trackData413_2.csv","w") as result:
        wtr= csv.writer( result )
        for r in rdr:
            del r[6:]
            wtr.writerow(( r) )
