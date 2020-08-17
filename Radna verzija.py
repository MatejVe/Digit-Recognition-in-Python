import pickle
import tkinter as tk
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#ucitavanje podataka
pickle_off = open('weights', 'rb')
weights = pickle.load(pickle_off)

pickle_off = open('biases', 'rb')
biases = pickle.load(pickle_off)


#tkinter
window = tk.Tk()

okvir = tk.Canvas(window)
okvir.config(bg='purple', height=280, width=280)
okvir.pack()

for i in range(28):
    ii = i + 1
    for j in range(28):
        jj = j + 1
        okvir.create_rectangle(i * 10, j * 10, ii * 10, jj * 10)

circles = {}
k = 0

lista = []
radius = 10


def move(event):
    global k, circles
    print(event.x, event.y)
    k += 1
    circles[k] = okvir.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='black')

    lista.append([event.x, event.y])


okvir.bind('<B1-Motion>', move)


def clean(event):
    global circles, k
    while k != 0:
        okvir.delete(circles[k])
        del circles[k]
        k -= 1
    #brisanje elemenata liste
    global lista
    lista = []

cleanButton = tk.Button(window)
cleanButton.config(text='Clean')
cleanButton.pack()

cleanButton.bind('<Button-1>', clean)


def recognise(event):

    matrica = []
    #bojanje malih pikselića
    for i in lista:
        for j in range(-5, 6):
            for z in range(-5, 6):
                if i[0] + j >= 0 and i[1] + z >= 0 and (i[0] + j, i[1] + z) not in matrica and i[0] + j < 280 and i[1] + z < 280:
                    matrica.append((i[0] + j, i[1] + z))


    data = np.zeros((784, 1))
    #popunjavanje zavrsne matrice
    for i, j in matrica:
        i2 = i//10
        j2 = j//10
        position = j2 * 28 + i2
        data[position][0] += 1

    a = data/100
    #računanje broja
    for i in range(2):
        a = sigmoid(np.dot(weights[i],a)+biases[i])
    print(a)
    print(np.argmax(a))

recogniseButton = tk.Button(window)
recogniseButton.config(text='Recognise')
recogniseButton.pack(side=tk.RIGHT)

recogniseButton.bind('<Button-1>', recognise)

window.mainloop()
