PREFIX = /usr/local
LIBDIR = $(PREFIX)/libexec/mafft
BINDIR = $(PREFIX)/bin
MANDIR = $(PREFIX)/share/man/man1
DESTDIR = 

#MNO_CYGWIN = -mno-cygwin

ENABLE_MULTITHREAD = -Denablemultithread
# Comment out the above line if your compiler 
# does not support TLS (thread-local strage).

#ENABLE_ATOMIC = -Denableatomic
# Comment out the above line if your compiler 
# does not support "atomic_int".

#DASH_CLIENT = dash_client
# Uncomment the above line to use protein 3D 
# structural information.  Go language is required.

CC = gcc
#CC = icc
CFLAGS = -O3
#CFLAGS = -O3 -fPIC
# add -fPIC when building .so files

#CC = icc
#CFLAGS = -fast
# if you have icc, use this.

#CFLAGS =  -O0  -fPIC -pedantic -Wall -std=c99 -g -pg -DMALLOC_CHECK_=3
#CFLAGS =  -O0  -fPIC -pedantic -Wall -std=c99 -g -pg -DMALLOC_CHECK_=3  -fprofile-arcs -ftest-coverage 
#CFLAGS =  -O0  -fPIC -pedantic -Wall -std=c99 -g -DMALLOC_CHECK_=3 # for shark, valgrind
#CFLAGS =  -O0  -fPIC -pedantic -Wall -std=c99 -g -DMALLOC_CHECK_=3 -lprofiler  # ?



ifdef ENABLE_MULTITHREAD
LIBS = -lm  -lpthread
else
LIBS = -lm
endif

ifdef ENABLE_ATOMIC
STDF = -std=c11
else
STDF = -std=c99
endif

MYCFLAGS = $(MNO_CYGWIN) $(ENABLE_MULTITHREAD) $(ENABLE_ATOMIC) $(STDF) $(CFLAGS)

INSTALL = install

STRIP = strip
#STRIP = true # to disable strip

PROGS = dvtditr dndfast7 dndblast sextet5 mafft-distance pairlocalalign \
		multi2hat3s pairash addsingle maffttext2hex hex2maffttext \
        splittbfast disttbfast tbfast nodepair mafft-profile f2cl mccaskillwrap contrafoldwrap countlen \
		seq2regtable regtable2seq score getlag dndpre setcore filter replaceu restoreu setdirection makedirectionlist version \
		$(DASH_CLIENT)
SOS = libdisttbfast.so
DLLS = libdisttbfast.dll
DYLIBS = libdisttbfast.dylib

PERLPROGS = mafftash_premafft.pl seekquencer_premafft.pl
SCRIPTS = mafft mafft-homologs.rb mafft-sparsecore.rb
OBJSETDIRECTION = mtxutl.o io.o setdirection.o defs.o mltaln9.o Galign11.o Lalign11.o genalign11.o
OBJFILTER = mtxutl.o io.o filter.o defs.o mltaln9.o Galign11.o Lalign11.o genalign11.o
OBJREPLACEU = mtxutl.o io.o replaceu.o defs.o mltaln9.o Galign11.o Lalign11.o genalign11.o
OBJRESTOREU = mtxutl.o io.o restoreu.o defs.o mltaln9.o Galign11.o Lalign11.o genalign11.o
OBJREGTABLE2SEQ = mtxutl.o io.o regtable2seq.o defs.o mltaln9.o Galign11.o Lalign11.o genalign11.o
OBJSEQ2REGTABLE = mtxutl.o io.o seq2regtable.o defs.o
OBJCOUNTLEN = mtxutl.o io.o countlen.o defs.o
OBJF2CL = mtxutl.o io.o f2cl.o constants.o defs.o
OBJMCCASKILLWRAP = mtxutl.o io.o mccaskillwrap.o constants.o defs.o mltaln9.o Galign11.o Lalign11.o genalign11.o
OBJCONTRAFOLDWRAP = mtxutl.o io.o contrafoldwrap.o constants.o defs.o mltaln9.o Galign11.o Lalign11.o genalign11.o
OBJMULTI2HAT3S = mtxutl.o io.o mltaln9.o tddis.o constants.o \
				 multi2hat3s.o defs.o fft.o fftFunctions.o Galign11.o Lalign11.o genalign11.o
OBJPAIRASH = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
			 Falign.o MSalignmm.o Galign11.o Lalign11.o genalign11.o MSalign11.o suboptalign11.o SAalignmm.o \
			 pairash.o defs.o fft.o fftFunctions.o
OBJPAIRLOCALALIGN = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		            Falign.o MSalignmm.o Galign11.o Lalign11.o genalign11.o MSalign11.o suboptalign11.o SAalignmm.o \
					pairlocalalignmain.o pairlocalalign.o defs.o fft.o fftFunctions.o
OBJDUMMY = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o \
			disttbfast_dummy.o dummy.o defs.o fft.o fftFunctions.o
OBJSPLITFROMALN = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o  Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o \
			splitfromaln.o defs.o fft.o fftFunctions.o
OBJSPLITTBFAST = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o  Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o \
			splittbfast.o defs.o fft.o fftFunctions.o 
OBJSPLITTBFASTP = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o \
			defs.o fft.o fftFunctions.o
OBJDISTTBFAST = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o \
			disttbfast.o defs.o fft.o fftFunctions.o addfunctions.o
OBJMAKEDIRECTIONLIST = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o  Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o \
			makedirectionlist.o defs.o fft.o fftFunctions.o addfunctions.o
OBJTBFAST = mtxutl.o io.o mltaln9.o tddis.o constants.o MSalignmm.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o \
			tbfast.o defs.o fft.o fftFunctions.o addfunctions.o \
			pairlocalalign.o MSalign11.o 
OBJNODPAIR = mtxutl.o io.o mltaln9.o tddis.o constants.o MSalignmm.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o \
			nodepair.o defs.o fft.o fftFunctions.o addfunctions.o \
			pairlocalalign.o MSalign11.o
OBJADDSINGLE = mtxutl.o io.o mltaln9.o tddis.o constants.o MSalignmm.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o \
			addsingle.o defs.o fft.o fftFunctions.o addfunctions.o
OBJSETCORE = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o \
			setcore.o defs.o fft.o fftFunctions.o
OBJTDITR = mtxutl.o io.o mltaln9.o tddis.o constants.o nj.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o  \
		   Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o fftFunctions.o fft.o  \
		   tditeration.o tditr.o defs.o SAalignmm.o treeOperation.o
OBJDVTDITR = mtxutl.o io.o mltaln9.o tddis.o constants.o nj.o partSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
			Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o MSalignmm.o fftFunctions.o fft.o  \
			tditeration.o dvtditr.o defs.o SAalignmm.o treeOperation.o addfunctions.o 
OBJGETLAG = mtxutl.o io.o mltaln9.o tddis.o constants.o partSalignmm.o Lalignmm.o rna.o Salignmm.o  \
		    Falign.o Falign_localhom.o Galign11.o Lalign11.o genalign11.o SAalignmm.o MSalignmm.o Dalignmm.o \
			getlag.o defs.o fft.o fftFunctions.o
OBJGAPFILL = mtxutl.o io.o constants.o gapfill.o defs.o
OBJDNDFAST7 = dndfast7.o io.o constants.o mtxutl.o mltaln9.o tddis.o defs.o Galign11.o Lalign11.o genalign11.o
OBJDNDBLAST = dndblast.o io.o constants.o mtxutl.o mltaln9.o tddis.o defs.o Galign11.o Lalign11.o genalign11.o
OBJSEXTET5 = io.o constants.o mtxutl.o mltaln9.o tddis.o sextet5.o defs.o Galign11.o Lalign11.o genalign11.o
OBJDISTANCE = io.o constants.o mtxutl.o mltaln9.o tddis.o mafft-distance.o defs.o Galign11.o Lalign11.o genalign11.o
OBJTRIPLET6 = io.o constants.o mtxutl.o mltaln9.o tddis.o triplet6.o defs.o
OBJTRIPLET5 = io.o constants.o mtxutl.o mltaln9.o tddis.o triplet5.o defs.o
OBJOCTET4 = io.o constants.o mtxutl.o mltaln9.o tddis.o octet4.o defs.o
OBJDNDPRE = dndpre.o io.o constants.o mtxutl.o mltaln9.o defs.o Galign11.o Lalign11.o genalign11.o
OBJGALN = io.o mtxutl.o mltaln9.o tddis.o constants.o partSalignmm.o MSalignmm.o Lalignmm.o rna.o Salignmm.o Dalignmm.o \
          SAalignmm.o Galign11.o Lalign11.o genalign11.o Falign.o Falign_localhom.o fftFunctions.o fft.o mafft-profile.o defs.o
OBJSCORE = io.o mtxutl.o mltaln9.o score.o constants.o defs.o Galign11.o Lalign11.o genalign11.o

HEADER = mltaln.h mtxutl.h mafft.h
FFTHEADER = fft.h

MANPAGES = mafft.1 mafft-homologs.1




all : $(PERLPROGS) $(PROGS) $(SCRIPTS)
	cp $(SCRIPTS) ../scripts
	chmod 755 ../scripts/*
	cp $(PERLPROGS) $(PROGS) ../binaries
	chmod 755 ../binaries/*
#	cp $(MANPAGES) ../binaries
	@echo done.

sos : $(SOS)
dylibs : $(DYLIBS)
dlls : $(DLLS)

$(DASH_CLIENT): dash_client.go
#	go build dash_client.go
	env CGO_ENABLED=0 go build --ldflags '-extldflags "-static"' dash_client.go # for conda


univscript: univscript.tmpl Makefile
	sed "s:_PROGS:$(PROGS):" univscript.tmpl  > univscript

mafft: mafft.tmpl mltaln.h
	sed "s:_LIBDIR:$(LIBDIR):" mafft.tmpl  > mafft

mafft-homologs.rb: mafft-homologs.tmpl
#	cp mafft-homologs.tmpl mafft-homologs.rb
	sed "s:_BINDIR:$(BINDIR):" mafft-homologs.tmpl  > mafft-homologs.rb

mafft-sparsecore.rb: mafft-sparsecore.tmpl
#	cp mafft-sparsecore.tmpl mafft-sparsecore.rb
	sed "s:_BINDIR:$(BINDIR):" mafft-sparsecore.tmpl  > mafft-sparsecore.rb

mltaln.h : functions.h
	touch mltaln.h

version : version.c mltaln.h
	$(CC) -o $@ version.c $(MYCFLAGS) $(LDFLAGS) $(LIBS) 

maffttext2hex : maffttext2hex.c
	$(CC) -o $@ maffttext2hex.c $(MYCFLAGS) $(LDFLAGS) $(LIBS)

hex2maffttext : hex2maffttext.c
	$(CC) -o $@ hex2maffttext.c $(MYCFLAGS) $(LDFLAGS) $(LIBS)

tbfast : $(OBJTBFAST)
	$(CC) -o $@ $(OBJTBFAST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

nodepair : $(OBJNODPAIR)
	$(CC) -o $@ $(OBJNODPAIR) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

addsingle : $(OBJADDSINGLE)
	$(CC) -o $@ $(OBJADDSINGLE) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

disttbfast : $(OBJDISTTBFAST)
	$(CC) -o $@ $(OBJDISTTBFAST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

libdisttbfast.so : $(OBJDISTTBFAST)
	$(CC) -shared -o $@ $(OBJDISTTBFAST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

libdisttbfast.dylib : $(OBJDISTTBFAST)
	$(CC) -dynamiclib -o $@ $(OBJDISTTBFAST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

libdisttbfast.dll : $(OBJDISTTBFAST)
	$(CC) -shared -o $@ $(OBJDISTTBFAST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

makedirectionlist : $(OBJMAKEDIRECTIONLIST)
	$(CC) -o $@ $(OBJMAKEDIRECTIONLIST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

splittbfast : $(OBJSPLITTBFAST)
	$(CC) -o $@ $(OBJSPLITTBFAST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

splitfromaln : $(OBJSPLITFROMALN)
	$(CC) -o $@ $(OBJSPLITFROMALN) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

dummy : $(OBJDUMMY)
	$(CC) -o $@ $(OBJDUMMY) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

setcore : $(OBJSETCORE)
	$(CC) -o $@ $(OBJSETCORE) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

countlen : $(OBJCOUNTLEN)
	$(CC) -o $@ $(OBJCOUNTLEN) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

seq2regtable : $(OBJSEQ2REGTABLE)
	$(CC) -o $@ $(OBJSEQ2REGTABLE) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

regtable2seq : $(OBJREGTABLE2SEQ)
	$(CC) -o $@ $(OBJREGTABLE2SEQ) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

setdirection : $(OBJSETDIRECTION)
	$(CC) -o $@ $(OBJSETDIRECTION) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

replaceu : $(OBJREPLACEU)
	$(CC) -o $@ $(OBJREPLACEU) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

filter : $(OBJFILTER)
	$(CC) -o $@ $(OBJFILTER) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

restoreu : $(OBJRESTOREU)
	$(CC) -o $@ $(OBJRESTOREU) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

f2cl : $(OBJF2CL)
	$(CC) -o $@ $(OBJF2CL) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

mccaskillwrap : $(OBJMCCASKILLWRAP)
	$(CC) -o $@ $(OBJMCCASKILLWRAP) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

contrafoldwrap : $(OBJCONTRAFOLDWRAP)
	$(CC) -o $@ $(OBJCONTRAFOLDWRAP) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

pairlocalalign : $(OBJPAIRLOCALALIGN)
	$(CC) -o $@ $(OBJPAIRLOCALALIGN) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

pairash : $(OBJPAIRASH)
	$(CC) -o $@ $(OBJPAIRASH) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

multi2hat3s : $(OBJMULTI2HAT3S)
	$(CC) -o $@ $(OBJMULTI2HAT3S) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

getlag : $(OBJGETLAG)
	$(CC) -o $@ $(OBJGETLAG) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

tditr : $(OBJTDITR)
	$(CC) -o $@ $(OBJTDITR) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

dvtditr : $(OBJDVTDITR)
	$(CC) -o $@ $(OBJDVTDITR) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

mafft-profile : $(OBJGALN)
	$(CC) -o $@ $(OBJGALN) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

gapfill : $(OBJGAPFILL)
	$(CC) -o $@ $(OBJGAPFILL) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

dndfast7 : $(OBJDNDFAST7)
	$(CC) -o $@ $(OBJDNDFAST7) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

dndblast : $(OBJDNDBLAST)
	$(CC) -o $@ $(OBJDNDBLAST) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

triplet : $(OBJTRIPLET)
	$(CC) -o $@ $(OBJTRIPLET) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

triplet3 : $(OBJTRIPLET3)
	$(CC) -o $@ $(OBJTRIPLET3) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

sextet3 : $(OBJSEXTET3)
	$(CC) -o $@ $(OBJSEXTET3) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

sextet4 : $(OBJSEXTET4)
	$(CC) -o $@ $(OBJSEXTET4) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

sextet5 : $(OBJSEXTET5)
	$(CC) -o $@ $(OBJSEXTET5) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

mafft-distance : $(OBJDISTANCE)
	$(CC) -o $@ $(OBJDISTANCE) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

triplet5 : $(OBJTRIPLET5)
	$(CC) -o $@ $(OBJTRIPLET5) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

triplet6 : $(OBJTRIPLET6)
	$(CC) -o $@ $(OBJTRIPLET6) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

octet4 : $(OBJOCTET4)
	$(CC) -o $@ $(OBJOCTET4) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

dndpre : $(OBJDNDPRE)
	$(CC) -o $@ $(OBJDNDPRE) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

score : $(OBJSCORE)
	$(CC) -o $@ $(OBJSCORE) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

genMtx : $(OBJGENMTX)
	$(CC) -o $@ $(OBJGENMTX) $(MYCFLAGS) $(LDFLAGS) $(LIBS)

mafftash_premafft.pl : mafftash_premafft.tmpl
	cp mafftash_premafft.tmpl mafftash_premafft.pl

seekquencer_premafft.pl : seekquencer_premafft.tmpl
	cp seekquencer_premafft.tmpl seekquencer_premafft.pl

gapfill.o : gapfill.c $(HEADER)
	$(CC) $(MYCFLAGS) -c gapfill.c

mltaln9.o : mltaln9.c $(HEADER)
	$(CC) $(MYCFLAGS) -c mltaln9.c

tddis.o : tddis.c $(HEADER)
	$(CC) $(MYCFLAGS) -c tddis.c

constants.o : constants.c miyata.h miyata5.h blosum.c DNA.h JTT.c $(HEADER)
	$(CC) $(MYCFLAGS) -c constants.c

defs.o : defs.c 
	$(CC) $(MYCFLAGS) -c defs.c

#A+++alignmm.o : SA+++alignmm.c $(HEADER)
#	$(CC) $(MYCFLAGS) -c SA+++alignmm.c -o A+++alignmm.o

Salignmm.o : Salignmm.c $(HEADER)
	$(CC) $(MYCFLAGS) -c Salignmm.c 

Dalignmm.o : Dalignmm.c $(HEADER)
	$(CC) $(MYCFLAGS) -c Dalignmm.c 

MSalignmm.o : MSalignmm.c $(HEADER)
	$(CC) $(MYCFLAGS) -c MSalignmm.c 

partSalignmm.o : partSalignmm.c $(HEADER)
	$(CC) $(MYCFLAGS) -c partSalignmm.c  

Lalign11.o : Lalign11.c $(HEADER)
	$(CC) $(MYCFLAGS) -c Lalign11.c 

genalign11.o : genalign11.c $(HEADER)
	$(CC) $(MYCFLAGS) -c genalign11.c 

suboptalign11.o : suboptalign11.c $(HEADER)
	$(CC) $(MYCFLAGS) -c suboptalign11.c 

Galign11.o : Galign11.c $(HEADER)
	$(CC) $(MYCFLAGS) -c Galign11.c 

MSalign11.o : MSalign11.c $(HEADER)
	$(CC) $(MYCFLAGS) -c MSalign11.c 

SAalignmm.o : SAalignmm.c $(HEADER)
	$(CC) $(MYCFLAGS) -c SAalignmm.c -o SAalignmm.o

Lalignmm.o : Lalignmm.c $(HEADER)
	$(CC) $(MYCFLAGS) -c Lalignmm.c 

rna.o : rna.c $(HEADER)
	$(CC) $(MYCFLAGS) -c rna.c 

disttbfast.o : disttbfast.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c disttbfast.c

splitfromaln.o : splitfromaln.c $(HEADER) $(FFTHEADER) 
	$(CC) $(MYCFLAGS) -c splitfromaln.c

splittbfast.o : splittbfast.c $(HEADER) $(FFTHEADER) 
	$(CC) $(MYCFLAGS) -c splittbfast.c

splittbfast2.o : splittbfast2.c $(HEADER) $(FFTHEADER) 
	$(CC) $(MYCFLAGS) -c splittbfast2.c

makedirectionlist.o : makedirectionlist.c $(HEADER) $(FFTHEADER) 
	$(CC) $(MYCFLAGS) -c makedirectionlist.c

disttbfast_dummy.o : disttbfast_dummy.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c disttbfast_dummy.c

dummy.o : dummy.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c dummy.c

tbfast.o : tbfast.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c tbfast.c

nodepair.o : nodepair.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c nodepair.c

addsingle.o : addsingle.c $(HEADER) $(FFTHEADER)
addsingle.o : addsingle.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c addsingle.c

tbfast2.o : tbfast2.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c tbfast2.c

setcore.o : setcore.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c setcore.c

getlag.o : getlag.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c getlag.c

tditr.o : tditr.c $(HEADER)
	 $(CC) $(MYCFLAGS) -c tditr.c

dvtditr.o : dvtditr.c $(HEADER)
	 $(CC) $(MYCFLAGS) -c dvtditr.c

tditeration.o : tditeration.c $(HEADER)
	 $(CC) $(MYCFLAGS) -c tditeration.c

mafft-profile.o : mafft-profile.c $(HEADER) $(MTXHEADER)
	$(CC) $(MYCFLAGS) -c mafft-profile.c

dndfast7.o : dndfast7.c $(HEADER) $(MTXHEADER)
	$(CC) $(MYCFLAGS) -c dndfast7.c

dndpre.o : dndpre.c $(HEADER)
	$(CC) $(MYCFLAGS) -c dndpre.c

countlen.o : countlen.c $(HEADER)
	$(CC) $(MYCFLAGS) -c countlen.c

seq2regtable.o : seq2regtable.c $(HEADER)
	$(CC) $(MYCFLAGS) -c seq2regtable.c

regtable2seq.o : regtable2seq.c $(HEADER)
	$(CC) $(MYCFLAGS) -c regtable2seq.c

f2cl.o : f2cl.c $(HEADER)
	$(CC) $(MYCFLAGS) -c f2cl.c

setdirection.o : setdirection.c $(HEADER)
	$(CC) $(MYCFLAGS) -c setdirection.c

replaceu.o : replaceu.c $(HEADER)
	$(CC) $(MYCFLAGS) -c replaceu.c

restoreu.o : restoreu.c $(HEADER)
	$(CC) $(MYCFLAGS) -c restoreu.c

mccaskillwrap.o : mccaskillwrap.c $(HEADER)
	$(CC) $(MYCFLAGS) -c mccaskillwrap.c

contrafoldwrap.o : contrafoldwrap.c $(HEADER)
	$(CC) $(MYCFLAGS) -c contrafoldwrap.c

pairlocalalign.o : pairlocalalign.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c pairlocalalign.c

pairlocalalignmain.o : pairlocalalignmain.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c pairlocalalignmain.c

pairash.o : pairash.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c pairash.c

multi2hat3s.o : multi2hat3s.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c multi2hat3s.c

io.o : io.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c io.c

nj.o : nj.c $(HEADER)
	$(CC) $(MYCFLAGS) -c nj.c

treeOperation.o : treeOperation.c $(HEADER)
	$(CC) $(MYCFLAGS) -c treeOperation.c

sextet5.o : sextet5.c $(HEADER) $(MTXHEADER)
	$(CC) $(MYCFLAGS) -c sextet5.c

mafft-distance.o : mafft-distance.c $(HEADER) $(MTXHEADER)
	$(CC) $(MYCFLAGS) -c mafft-distance.c

maffttext2hex.o : maffttext2hex.c
	$(CC) $(MYCFLAGS) -c maffttext2hex.c

hex2maffttext.o : hex2maffttext.c
	$(CC) $(MYCFLAGS) -c hex2maffttext.c

triplet6.o : triplet6.c $(HEADER) $(MTXHEADER)
	$(CC) $(MYCFLAGS) -c triplet6.c

fft.o : fft.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c fft.c 

fftFunctions.o : fftFunctions.c $(HEADER) $(FFTHEADER)
	$(CC) $(MYCFLAGS) -c fftFunctions.c

Falign.o : Falign.c $(HEADER) $(FFTHEADER) $(MTXHEADER)
	$(CC) $(MYCFLAGS) -c Falign.c

Falign_localhom.o : Falign_localhom.c $(HEADER) $(FFTHEADER) $(MTXHEADER)
	$(CC) $(MYCFLAGS) -c Falign_localhom.c

mtxutl.o : mtxutl.c 
	$(CC) $(MYCFLAGS) -c mtxutl.c

addfunctions.o : addfunctions.c $(HEADER)
	$(CC) $(MYCFLAGS) -c addfunctions.c

score.o : score.c $(HEADER)
	$(CC) $(MYCFLAGS) -c score.c

clean :
	rm -f *.o *.a *.exe *~ $(PERLPROGS) $(PROGS) $(SCRIPTS) $(SOS) $(DYLIBS) $(DLLS) *.gcda *.gcno $(DASH_CLIENT)
#	rm -f ../binaries/* ../scripts/*

install : all
	mkdir -p $(DESTDIR)$(LIBDIR)
	chmod 755 $(DESTDIR)$(LIBDIR)
	mkdir -p $(DESTDIR)$(BINDIR)
	chmod 755 $(DESTDIR)$(BINDIR)
	chmod 755 $(SCRIPTS)
	$(INSTALL) $(SCRIPTS)  $(DESTDIR)$(BINDIR)
	chmod 755 $(PROGS) ||:     # in MinGW, it's ok if this fails
#	$(INSTALL) -s $(PROGS) $(DESTDIR)$(LIBDIR)
	$(STRIP) $(PROGS) ||: # may fail for dash_client on mac.
	$(INSTALL) $(PROGS) $(DESTDIR)$(LIBDIR)
	$(INSTALL) $(PERLPROGS) $(DESTDIR)$(LIBDIR)
#	$(INSTALL) -m 644 $(MANPAGES) $(DESTDIR)$(LIBDIR)

	( cd $(DESTDIR)$(BINDIR); \
rm -f linsi ginsi einsi fftns fftnsi nwns nwnsi xinsi qinsi; \
rm -f mafft-linsi mafft-ginsi mafft-einsi mafft-fftns mafft-fftnsi mafft-nwns mafft-nwnsi mafft-xinsi mafft-qinsi mafft-randomcore.rb ; \
ln -s mafft linsi; ln -s mafft ginsi; ln -s mafft fftns; \
ln -s mafft fftnsi; ln -s mafft nwns; ln -s mafft nwnsi; \
ln -s mafft einsi; \
ln -s mafft mafft-linsi; ln -s mafft mafft-ginsi; ln -s mafft mafft-fftns; \
ln -s mafft mafft-fftnsi; ln -s mafft mafft-nwns; ln -s mafft mafft-nwnsi; \
ln -s mafft mafft-einsi; ln -s mafft mafft-xinsi; ln -s mafft mafft-qinsi;\
rm -f mafft-profile mafft-profile.exe; ln -s $(LIBDIR)/mafft-profile .; \
rm -f mafft-distance mafft-distance.exe; ln -s $(LIBDIR)/mafft-distance . )

	mkdir -p $(DESTDIR)$(MANDIR)
	chmod 755 $(DESTDIR)$(MANDIR)
	$(INSTALL) -m 644 $(MANPAGES) $(DESTDIR)$(MANDIR)
# remove incorrectly installed manpages by previous versions
#	rm -f /usr/local/man/man1/mafft.1 /usr/local/man/man1/mafft-homologs.1 
