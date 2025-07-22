import csv
import pandas as pd

df = pd.read_csv("result-release_rate/energies_debonded2D.csv")

filename = 'result-release_rate/energies_debonded2D'

ax = df.plot(x='$\delta$', y=['Theoretical $d=1.62$mm', 'Energy $d=1.62$mm',\
                           'Theoretical $d=3.00$mm', 'Energy $d=3.00$mm'], fontsize=15)
ax.set_xlabel('Bonded fraction $\delta$', fontdict={'fontsize':16})
ax.set_ylabel('Energy per unit width [mJ/mm]', fontdict={'fontsize':17})
ax.legend(prop={'size': 12})
ax.figure.savefig(filename + '1.eps')

ax = df.plot(x='$\delta$', y=['Theoretical $d=5.00$mm', 'Energy $d=5.00$mm',\
                           'Theoretical $d=15.00$mm', 'Energy $d=15.00$mm'], fontsize=15)
ax.set_xlabel('Bonded fraction $\delta$', fontdict={'fontsize':16})
ax.set_ylabel('Energy per unit width [mJ/mm]', fontdict={'fontsize':17})
ax.legend(prop={'size': 11})
ax.figure.savefig(filename+'2.eps')

print('Plot saved at ', filename)