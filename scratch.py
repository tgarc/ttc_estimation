dfile= open('run1.txt','r')

while(1):
    startpos = dfile.tell()
    l = dfile.readline()
    if l[0] != '#': break
    comments = l[1:].strip()
    
dfile.seek(startpos)

data = [d.strip().split(',') for d in dfile]
types = data[0]
names = data[1]
cast = lambda x: [eval(t + '('+d+')') for t,d in zip(types,x)]
cdata = np.array([cast(d) for d in data[2:]])
for col, na in enumerate(names):
    locals()[na] = cdata[:,col]

clf()
thigh = np.sort(ttc-np.mean(ttc))[0.95*len(ttc)]
plot(frame, ttc, 'b-o')
plot(frame[ttc>thigh], ttc[ttc>thigh],'r*',markersize=10)
plot(frame[ttc<=thigh], ttc[ttc<=thigh] - 10,'g--o',linewidth=1.5,markersize=3)
grid()
# plot(frame, -frame + len(frame),'--',color='orange')
