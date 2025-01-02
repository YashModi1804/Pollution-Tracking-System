There have been a few changes made to implement and run the shapefile feature:

1)app.py (lines 25-38): Only this section has been altered. The previous API connection has been bypassed.
2)Home.html: 90% of the work has been done on this part.
3)Shapefiles: The remaining 10% includes the shapefiles stored in the static folder under dissolved_output. The internal features of the shapefiles have been removed.
4)Pseudo creds2.json file: This file is used solely for bypassing the backend during testing.
5)Openstreet map been replaced by Bhuvan Map.