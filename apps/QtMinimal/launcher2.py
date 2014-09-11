import tkinter
import subprocess

#class QtMinimalXBARLauncher(tkinter.Frame):
#    def __init__(self, master=None):
#        Frame._init__(self, master)
#        self.master.title("QtMinimalXBARLauncher")

choices = list()
listboxes = dict()
choices.append( ("filename",
                 ["c:\\models\\icido\\body.nbf",
                  "e:\\tmp\\qtminimalxbar.nbf",
				  "t:\\models\\nasa\\f35a_texture_noDOFs\\cloned.xml",
				  "t:\\models\\nasa\\f35a\\cloned.xml",
				  "e:\\models\\icido\\bombardier.nbf",
                  "e:\\models\\ds\\SEAT_wedges_unoptim.nbf",
                  "e:\\models\\ds\\ISD_CAR_wedges_unoptim.nbf",
                  "e:\\models\\ds\\JET_woedges_unoptim.nbf",
                  "e:\\models\\ds\\JET_wedges_unoptim.nbf",
                  "e:\\models\\ds\\JET_woedges_optim.nbf",
                  "e:\\tmp\\test.nbf",
                  "cubes",
				  "t:\\models\\maxbench\\textured1.nbf",
                  "\"E:\\models\\HOOPS\\inventor\\Digger Benchmark\\Digger\\Workgroups\\NotInProject\\Heavy Duty Benchmark.iam\"",
                  "\"E:\\models\\HOOPS\\SolidWorks\\Vermeer 2010\\#HG 365-FULL.SLDASM\""
                  ] ) )
choices.append( ("build", ["Debug", "RelWithDebInfo", "Release"]) )
choices.append( ("renderengine", ["VAB", "VBO", "VBOVAO", "Bindless", "BindlessVAO", "DisplayList", "Indirect"]) )
choices.append( ("shadermanager", ["rixfx:uniform", "rixfx:shaderbufferload", "rixfx:ubo140", "rixfx:ssbo140"]) )

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
    command = command + " --headlight"
    command = command + " --combineVertexAttributes"
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
