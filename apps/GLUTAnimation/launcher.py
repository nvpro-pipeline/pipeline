import tkinter
import subprocess

#class QtMinimalXBARLauncher(tkinter.Frame):
#    def __init__(self, master=None):
#        Frame._init__(self, master)
#        self.master.title("QtMinimalXBARLauncher")

choices = list()
listboxes = dict()
choices.append( ("filename", ["c:\\models\\icido\\body.nbf",]) )
choices.append( ("build", ["Debug", "RelWithDebInfo", "Release"]) )
choices.append( ("renderengine", ["VBO", "VBOVAO", "Bindless", "BindlessVAO", "DisplayList"]) )
choices.append( ("shadermanager", ["uniform", "ubo", "shaderbuffer"]) )

selections = dict()
                 
parameters = list()

root = tkinter.Tk()

class ListUpdater():
    def __init__( self, name ):
        self.name = name

    def update( self, event ):
        print(event)
        widget = event.widget
        index = int(widget.curselection()[0])
        selections[self.name] = widget.get(index)
        print(selections)

def launch():
    command = "F:\\perforce\\sw\\wsapps\\nvsgsdk\\nvsgmain\\bin\\amd64\\win\\crt100\\" + selections["build"] + "\\QtMinimalXBAR.exe --continuous"
    command = command + " --renderengine " + selections["renderengine"]
    command = command + " --shadermanager " + selections["shadermanager"]
    command = command + " --filename " + selections["filename"]
    print(command)
    subprocess.call(command)

index = 1;
for (parameter, values) in choices:
    print(parameter)
    listbox = tkinter.Listbox(root)
    listboxes[parameter] = listbox
    listbox.bind('<<ListboxSelect>>', ListUpdater(parameter).update )
    
    selections[parameter] = values[0]
    for item in values:
        listbox.insert(tkinter.END, item)

    listbox.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
       

button = tkinter.Button( root, text="Launch", command=launch )
button.pack( side=tkinter.BOTTOM )
print(parameters)

root.geometry("800x300+10+10")
root.mainloop()
