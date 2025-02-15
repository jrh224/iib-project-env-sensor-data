## 9th Oct: Investigating the external weather data sent to me by Tash

from matplotlib import pyplot as plt
from utils.CustomDataframe import CustomDataframe

combinedData = CustomDataframe(filename="DataSorted_2024-08-08-09-09__SENS_36DC40metdata_combined.csv")

combinedData.filter_by_date(days=5)

combinedData.plot("T", label="Internal Temp")
combinedData.plot("temp", label="External Temp")

plt.title("Internal vs External Room Temperature plotted for DataSorted_2024-08-08-09-09__SENS_36DC40metdata_combined")
plt.ylabel("Temperature, C")

plt.show()