# DATASETS
From the paper:
>  We used three different datasets to evaluate the performances of our method.
>  The first dataset (``Mozart'') consists of $38$ movements from (13) Mozart Piano Sonatas, for which the main melody line was annotated manually by a professional pianist.
>  The second dataset (``Pop'') consists of $83$ popular songs (including pop and jazz).
>  We used the vocal part of these songs as the melody line, and treated them as though they were compressed onto a single track (identifying the main track in multi-track music is a separate question, see~\cite{HUANG2010,Martin2009, Li2009, Friberg2009}).
>  These datasets were used for training and testing.
>  A third dataset (``Web''), used only for testing, comprises MIDI files crawled from the web.
>  This dataset includes $169$ Western art music compositions from the late 16th to the early 20th centuries.
>  All of these pieces included a solo instrument (typically voice, flute, violin or clarinet) and accompaniment (typically strings or piano).
>  The first and third of these datasets are publicly available for research purposes in the companion site -- see footnote 1.
>  We do not have distribution rights for the second dataset, which was professionally curated and annotated, but we provide the full list of pieces.

The full list of the pieces contained in the datasets is available here, alongside to the
datasets themselves.

## Compressed archive
In the compressed archive you can find the pickled files. You can open them with the
function `data_handling.parse_data.load_piece` provided in this repository.

Pickled objects are much smaller and faster to be loaded than midi files.
Each pickled object is a list of notes and each note has the following fields:
* midi pitch
* offset
* duration
* soprano (1 if it is solo instrument note, 0 otherwise)

*offset* and *duration* are in the `music21` format:
- 1.0 -> 1/4 note
- 0.5 -> 1/8 note
- 2.0 -> half note
- etc.

The minimum duration is 1/32.

Inside the archive, there are three directories:

1. `mozart` with the Mozart piano sonatas
2. `validation` with the files used to validate models organized by authors
3. `cultivated` a larger dataset collected from the web from which the `validation` was
   extracted (not organized by authors)

In addition, the `pop` directory was the one with pop music but it was removed from this
public repository because of copyright issues.

## Download
Click [here](https://github.com/2019paper/Symbolic-Melody-Identification/raw/master/data/solo-accompaniment-dataset-pickled-nopop.tar.xz) to download the compressed archive.

## List of files
In 
[hyperopt.list](https://github.com/2019paper/Symbolic-Melody-Identification/blob/master/data/hyperopt.list)
you can find the list of files used for the hyper-parameter optimization, while in
[not_hyperopt.list](https://github.com/2019paper/Symbolic-Melody-Identification/blob/master/data/not_hyperopt.list)
are listed all the other files.

The full list of files of the datasets used is the following:

___

### CULTIVATED MUSIC ~1600-~1960
tot 233
```
BACH
BWV 1020
BWV 1025
BWV 1030
BWV 1031
BWV 1032
BWV 1033
BWV 1034
BWV 1035
BWV 1042 1st mov
BWV 1067

ALBENIZ
Tango tr guitar & flute

BEETHOVEN
Violin Sonata No 1, 1st mov
Adelaide
The Glory of God in Nature (The heavens declare) Op. 48 no. 4
I love you
Mailied

CORELLI
Sonata per violino n12 (Variationi sulla Follie d'Espagne, excerpts)

TARTINI
Sonate per violino: Re Maggiore

MOZART
Sonate fur Violin und Klavier KV 296, KV 376/380
The satisfaction
warning
Lullaby (Sleep, my little prince ...)
Abendempfindung
An Chloe
In early spring
Longing for the springs (Come dear May ...)
The violet

RAVEL
Pavane pur une infante defunte

SAINT-SAENS
Sonate pour Clarinet et Piano, Op. 167

VIVALDI
Concerto per flautino??

POULENC
Sonata for Clarinet and Piano FP 184 (no 3rd mov)

ROSSINI
Variazioni per Clarinetto e Pianoforte, IGR 72 

SCHUBERT
Sonate für Violine und Klavier A-Dur Op.162
After work Op. 25 no. 5
Ave Maria Op. 52 no. 6 (D 839)
An die Musik Op. 88 NO. 4
An der Laute On. 81 NO. 2
To a source Op. 109 no. 3
An Silvia Op. 106 Nr. 4
to sing on the water Op. 72
Thanksgiving to the Bach Op. 25 no. 4
The fisherman girl
Hiking Op. 25 no. 1
Die Forelle on. 32
The lookalike
Gretchen am Spinnrade Op. 2
Five Songs Op. 5
The hunter Op. 25 Nr. 14
Der Lindenbaum Op. 89 no. 5
There Mouse Sohn Op. 92 Number. 1
The Curious Op. 25 no. 6
The youth at the source, D300
Death and the Maiden Op. 7 no. 3
The butterfly Op. 57 no. 1
Der Wanderer an den Mond, Op. 80 no. 1
The miller flowers Op. 25 Nr. 9
The love color Op. 25 Nr. 16
Du bist die Ruh 'Op. 59 no. 3
Jealousy and pride Op. 25 Nr. 15
On Erlkönig. 1
Erntelied
spring song
Halt On. 25, Nr. 3
On Heideröslein. 3 No. 3
At sunset DV 799
Hunter Abendlied Op. 3. No. 4
Lover in all forms
Lied to Mignon, Op. 62 no. 4
My!
Minnelied 
With the beautiful sound ties Op. 25 Nr. 13
Morgengruss Op. 25 no. 8
Break Up. 25 no. 11
Peace be with you 
Bliss DV 433
On Schlummerlied. 24, Nr. 2
sigh
serenade
Daily sing
On Thränenregen. 25, Nr. 10
Impatience Op. 25 Nr. 7
Wanderers Night Song Op. 96 Nr. 3
On Heideröslein. 3 No. 3
Wiegenlied Op. 98 Nr. 2
Where? Op. 25 no. 2

LOEWE
The clock
Henry the Fowler
Prinz Eugen

HAYDN
A small house
Your Bells of Marling

LISZT
It must be a wonderful

MENDELSSOHN
Lied ohne Worte für Cello und Klavier Op.109
Gruss (Gently through my mind)
On Wings of Song

SCHUMANN
Romanzen für Oboe und Klavier Op.94 
Fünf Stücke im Volkston für Violoncello und Klavier Op.102
Märchenbilder für Viola und Klavier Op.113 
The Lotosblume on. 25 NO. 7
You're like a flower Op. 25 Nr. 24
First Green Op. 35 no. 4
Frühlingsfahrt Op. 45 no. 2
In the beautiful month of May, Op. 48 no. 1
From my tears sprout Op. 48 no. 2
The rose, the lily, the dove Op. 48 no. 3
When I look into your eyes Op. 48 no. 4
I bear no grudge Op. 48 no. 7
A young man loved a girl Op. 48 no. 11
Every night in my dreams Op. 48 no. 14
On Jasminenstrauch. 27, Nr. 4
Mondnacht Op. 39 Nr. 5 (Keyboard Song)
In the morning I get up and ask Op. 24 no. 21 (Klavierlied)
Volksliedchen Op. 51 Nr. 2 (Keyboard Song)
Wanderlied "Arise, and drank ..." Op. 35 no. 3 (Klavierlied)
Dedication Op. 25 no. 1 (piano song)

BRAHMS
Sonate für Violine und Klavier d-moll Op.108
Sonate für Violine und Klavier A-Dur Op.100
Sonate für Violine und Klavier G-Dur Op.78 
Of eternal love (dark, as dark)
On the lake, Op. 59 no. 2
The death, which is the cool night
May Night (When the silver moon)
Allow me feins girls
Feldeinsamkeit
In silent Night
My girl has a rose mouth
Sunday
Lullaby (Good evening, good night)

BUSONI
Sonata per violino e pianoforte in Mi Minore Op.29
Sonata per violino e pianoforte in Mi Minore Op. 36a (movimento 1)

REGER
Sechs Vortragsstücke für Violine und Klavier, Op.103a 
12 Vortragsstücke für Violine und Klavier
happiness

CHOPIN
Grand Vals Brilannte, flute and piano
The Maiden's wish

P. CORNELIUS
Unfaithful
Christmas tree
The Kings
Christ child

R. FRANZ
Good night! Op. 5 Nr. 7

GLUCK
The summer night

GRIEG
Solveig's Lied
To the Motherland, Op. 58 no. 2

GOUNOD
Ave Maria (Meditation)

F. REICHARDT
Blumengruß
Heidenröslein

J. DOWLAND
Three Lautenlieder
Stay Time awhile thy flying (Lautenlied)

SILCHER
Cute Little Anny
Lore-Ley

TELEMANN
the happiness

WAGNER
dreams

H. WOLF
Grab Anakreons
In an old picture
The abandoned maid
singing Weylas

K.F. ZELTER
There Mouse Sohn
The King of Thule

CM VON WEBER
Two guitar songs
Secret love pain Op. 64 no. 3
Wiegenlied Op. 13, Nr. 2
Variations for Clarinet and Piano, Op. 33

JOLLY:
FAE Sonate (Frei aber einsam) für Violine und Klavier - Dietrich, Schumann, Brahms, Schumann
(von 3 Komponisten für den gemeinsamen Freund, dem Geiger Joseph Joachim) 
```

___

### POP SONGS
tot 83
```
Another Love (Tom Odell) PVG                         
Man I Love The
Can_t Pretend (Tom Odell) PVG                        
Maybe
Concrete (Tom Odell) PVG                             
Mine
Daddy (Tom Odell) PVG                                
Momentum (Jamie Cullum) PVG
Entertainment (Tom Odell) PVG                        
Motion Picture Soundtrack (Radiohead) PVG
Everything In Its Right Place (Radiohead) PVG        
My Iron Lung (Radiohead) PVG
Everything You Didn_t Do (Jamie Cullum) PVG          
My One And Only
Exit Music For A Film (Radiohead) PVG                
Nice Work If You Can Get It
Fake Plastic Trees (Radiohead) PVG                   
No Surprises (Radiohead) PVG
Fascinatin_ Rhythm                                   
Of Thee I Sing (Baby)
Fog (Again) (Radiohead) PVG                          
Oh, Kay
Get Hold Of Yourself (Jamie Cullum) PVG              
Oh, Lady Be Good
Give me the simple life (Jamie Cullum)               
Punchup At A Wedding, A (Radiohead) PVG
Grow Old With Me (Tom Odell) PVG                     
Pyramid Song (Radiohead) PVG
Heal (Tom Odell) PVG                                 
Rosalie.mid
High _ Dry (Jamie Cullum)                            
Sad Sad World (Jamie Cullum) PVG
High _ Dry (Radiohead) PVG                           
Sail To The Moon (Radiohead) PVG
How I Made My Millions (Radiohead) PVG               
Silent Night (Tom Odell) PVG
How Long Has This Been Going On_                     
Slap That Bass
If I Ruled The World (Jamie Cullum) PVG              
Somebody Loves Me
I Got Rhythm                                         
Somehow (Tom Odell) PVG
I Know (Tom Odell) PVG                               
Someone To Watch Over Me
Isn’t It a Pity                                      
Sparrow (Tom Odell) PVG
I Thought I Knew What Love Was (Tom Odell) PVG       
Spending All My Christmas With You (Next Year) (Tom ODell) PVG
I’ve Got a Crush On You                              
Street Spirit (Radiohead) PVG
I Want None Of This (Radiohead)                      
Strike Up The Band
Jealousy (Tom Odell) PVG                             
Subterranean Homesick Alien (Radiohead) PVG
Just One Of Those Things (Jamie Cullum) PVG          
Supposed To Be (Tom Odell) PVG
Swanee
Karma Police (Radiohead)                             
Sweet And Low-down
Knives Out (Radiohead) PVG                           
Take Me Out (Jamie Cullum) PVG
Last flowers to the hospital                         
That Certain Feeling
Let’s Call The Whole Thing Off                       
The Half Of It, Dearie, Blues
Let’s Kiss And Make Up                               
They All Laughed
Life In A Glasshouse (Radiohead) PVG                 
They Can’t Take That Away From Me
Like Spinning Plates (live version) (Radiohead) PVG  
Things Are Looking Up
Long Way Down (Tom Odell) PVG                        
Till I lost (Tom Odell) PVG
Love Is Here To Stay                                 
Videotape (Radiohead) PVG
Love Is Sweeping The Country                         
We Suck Young Blood (Radiohead) PVG
Love Walked In                                       
When I Get Famous (Jamie Cullum) PVG
Lucky (Radiohead) PVG                                
Who Cares
Make Someone Happy (Jamie Cullum)                    
Wolf At The Door, A (Radiohead) PVG

```

___

### MOZART PIANO SONATA
tot: 38

N.B. KV 330, 2nd mov exists but is broken
```
KV 279, 1st mov
KV 279, 2nd mov
KV 279, 3nd mov
KV 280, 1st mov
KV 280, 2nd mov
KV 280, 3nd mov
KV 281, 1st mov
KV 281, 2nd mov
KV 281, 3nd mov
KV 282, 1st mov
KV 282, 2nd mov
KV 282, 3nd mov
KV 283, 1st mov
KV 283, 2nd mov
KV 283, 3nd mov
KV 284, 1st mov
KV 284, 2nd mov
KV 284, 3nd mov
KV 330, 1st mov
KV 330, 3nd mov
KV 331, 1st mov
KV 331, 2nd mov
KV 331, 3nd mov
KV 332, 1st mov
KV 332, 2nd mov
KV 332, 3nd mov
KV 333, 1st mov
KV 333, 2nd mov
KV 333, 3nd mov
KV 457, 1st mov
KV 457, 2nd mov
KV 457, 3nd mov
KV 475, 1st mov
KV 475, 2nd mov
KV 475, 3nd mov
KV 533, 1st mov
KV 533, 2nd mov
KV 533, 3nd mov
```
