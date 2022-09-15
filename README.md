# Who is Erik Hyrkas
I'm Erik Hyrkas. I'm a generalist. I code, but I also write science fiction and fantasy books, create video games, 
and thrive on creating and learning new things.

## MIT License 
Everything that is mine you are welcome to use. I heavily referenced https://github.com/Zeta36/chess-alpha-zero 
and https://colab.research.google.com/drive/1GSeBQdyZH_nHvl52XW0uhXV3Cslho24O, but I don't recall directly copying
and pasting any of their work. That said, you'll find some strong similarities because of how much I learned from them.

Examples of where I learned the most and where you'll find strong similarities with my references include
`get_uci_to_encoding_dict()` which is heavily based on work from Zeta36 and dramatically helped me encode moves better.
Everything in the `train_model.py` file is something that I learned from reading examples and is a mishmash of 
experimentation, but also learning from seeing what others did. I started out a lot closer to what was given by
https://www.youtube.com/watch?v=ffzvhe97J4Q and slowly evolved by looking at what others had done, including what 
Zeta36 did. The UCI portion of the engine was learned by looking at other UCI engines, and is half-baked, but adequate
for learning purposes.

I've tried to give credit in this document to everybody that I've learned from. Any website that I left out was an
oversight on my part and not malicious.

The end result is my own, for better or worse. I will go into more detail on my suspicions for possible improvements 
later in this README.  
## And why would you bother looking at this project?
I have done everything from application development to information security to cloud architecture to data engineering. 
I've built or rebuilt different technologies that exist to either make them better or to learn from them. Some of them
were created as paid work when these technologies were still sparkling and new and there was no clear dominate market
leader. Other technologies were well established, but I wanted a deeper understanding. I have built things like 
a search engine, a recommendation engine, a proxy server, a word processor with spell checker and some fun NLP tech, a 
web server, a web browser, a voice over internet protocol and application, a sql database, a distributed queue, a 
distributed cache, a dependency injection framework, an ORM framework, programing languages, scripting languages, 
game engines, games, database drivers, and other insanely tedious and difficult to build technologies, just for 
the sake of understanding them. What's more I love learning new languages. I started with BASIC in 1983, then 
x86 macro assembler in 1987, and have branched out to pretty much everything. There are many languages I 
haven't worked with and plenty that I haven't put in enough hours to be an expert with, but I've seen some stuff.

You've never heard of me, probably because nothing I've made was ever important enough or unique enough to be useful, 
or maybe it was because I was simply paid to build it and there is no public credit for such things, but maybe 
something that I'm learning about can help you.

## Why didn't you...
This project was about learning, not perfection or discipline. There are no unit tests. There is no model registry. 
There is no CI/CD process or MLOPs. You can think of this project as a doodle that came to life. I've been on a number of
professional ML projects where discipline was necessary for an operational model. This is not a professional project,
but rather a chance to experiment and learn in my own freeform way of trial and error.

## My Hardware and Software Setup
The only thing that really matters for this project is my video card. It's a couple of years old at this point, and I 
think it would have been handy to have a nicer card, but it worked well enough. 

In case you care, here are the relevant pieces:
* Video card: NVIDIA GeForce RTX2080 SUPER
* Disk: 1 TB solid state 
* Processor: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz   3.60 GHz
* Ram: 32 gb
* Python: 3.9
* OS: Windows 11 Home edition

I read somewhere that one of the reasons Stockfished shied away from incorporating certain types of ML (like what
AlphaZero did) was because they wanted the engine to work on general hardware. Sure, it works better with more
hardware, but it works. I didn't experiment with minimum requirements for using this chess engine, but I don't 
think it will work well on anything less that what I had.

# The LEG Chess Engine
My son Gavin asked me if we could make a chess engine (he's currently very passionate about chess), and my other son 
Logan wanted to help out and learn about machine learning. While I studied chess under Daniel Rensch before he was a 
big shot, my anxiety got in the way of me being able to compete over the board. That effort was probably the greatest
waste of his valuable time across his entire career. I did learn a lot, even if I wasn't able to overcome my anxiety
and play competitively. I still do puzzles every day to stay in shape, but I've never been a great player.

Anyway, with my sons' interests in mind, the LEG engine was born. L-E-G stands for Logan, Erik, and Gavin. 
Really, one of two obvious options for arranging our three names. We could have gone with the GEL engine, 
but here we are.

What's more, I flippantly told my boys that we could achieve a 1700 elo rating with our engine, "No problem." 
A few months later, and many, many hours of hard work, and the LEG engine did it. We had to train on
roughly 20 million unique board positions to get it to the point of playing at this skill level sometimes. 

Why is 1700 significant? In my mind, you don't beat a 1700 rated opponent with luck. You have to know how to play basic
chess. It's not that 1700 elo chess players are great, but there's virtually no way they'd lose to an opponent making
random moves. More than anything, I've studied a little chess and consider myself to be a casual player, and 1700 
felt like something that shouldn't be very hard to beat compared to 2100 or something that would be harder for me
personally to judge whether the engine was playing well. What I didn't realize was just how hard it is to make a 
chess engine that could beat a 1200 rated opponent, let alone a 1700 rate player. 

While the final model of this engine played Shredder at 2400 rating to a draw (Shredder even had the white pieces), 
I suspect that the engine is probably closer to 1500 elo without any additional tweaks, but it certainly can beat 
Stockfish 1700 fairly consistently over hundreds of games.

One of the most interesting things I've learned from this effort is what I call "the archer problem." (There's probably
a proper name for this phenomenon, but this is what's going on in my head.) A human who has never shot an arrow at a 
target might be pretty bad, but they will generally shoot in the direction of the target very early on in their 
learning. After a few hours of practice, most of the arrows will always be near the target and a few may hit. 
Eventually, all of their arrows will hit the target and some will even hit the bullseye. Give more time and more 
practice, they will have more and more arrows hit the bullseye. My experience with machine learning is that
the computer archer will initially shoot arrows in every direction, and after lots of practice it will start hitting
the bullseye more and more, but the arrows that miss could still be off in a random direction that isn't even near
the target. You might have 95% of all arrows get a bullseye and the other 5% are in your foot. There probably are
some great ways of mitigating this, but I think that the easiest answer is to have it train more.

Detecting bad predictions by using other (likely dubious) predictions, as it turns out, is hard. However, that's 
more or less what I did. I predict what the best moves might be, and then look over each of those potential best 
moves and predict whether the new position is good or bad. This doesn't mean that we don't shoot ourselves in the 
foot sometimes, but it's far less common.

## Why Machine Learning?
I had a fairly good idea on how a heuristic chess engine might work, and I wanted to learn something
new, and besides, Logan wanted to learn about machine learning as well. I've been part of a number of machine learning
projects both professionally and for my own fun, but there was a lot I would need to learn to make a chess engine.

## Why reinvent something?
Chess engines exist, some of which are incredibly powerful. The LEG engine doesn't really add anything
new to computer science or chess, but it did teach me a lot and I hope it teaches my sons something 
as well. If you learn something, then this will have all been worthwhile.

## What did I learn?
The code I checked into GitHub is my 3rd attempt. You won't see all of my previous failures, but let me assure you
that they were epic.

I read about Micro-Max (https://home.hccnet.nl/h.g.muller/max-src2.html) and Sunfish 
(https://github.com/thomasahle/sunfish), all about Alpha-Beta pruning 
(https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning), Monte Carlo Tree Search 
(https://en.wikipedia.org/wiki/Monte_Carlo_tree_search), Python Chess 
(https://python-chess.readthedocs.io/en/latest/engine.html), UCI protocol (http://wbec-ridderkerk.nl/html/UCIProtocol.html),
and FENs (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation). A good start, but I wanted to know more 
about Machine Learning.

I read more about efforts around machine learning and chess:
* https://web.stanford.edu/~surag/posts/alphazero.html
* https://en.wikipedia.org/wiki/Leela_Chess_Zero
* https://www.chessprogramming.org/Stockfish_NNUE

I read about convolutional neural networks:
* https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
* https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/

I read about dense layers :
* https://machinelearningknowledge.ai/keras-dense-layer-explained-for-beginners/

I read more about why my neural network wasn't working:
* https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

I spent a lot of time watching videos by Andrew Ng (This guy is amazing!):
* https://www.youtube.com/playlist?list=PL1jfcDaRop0Ji3cBq8P1sn-cBiG_ReMdK
* https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0

I watched a YouTube video on making a chess engine with Tensor Flow: https://www.youtube.com/watch?v=ffzvhe97J4Q

His example attempted to predict the centipawn evaluation that Stockfish would give a position and use that to 
find the best move. It was educational, but the engine wasn't very good. I ran the code locally and I don't think it 
was even 300 ELO. It was just slightly better than making random moves, but it gave me a great starting point that let 
me learn from there. The greatest disservice this video did me was the impression of how much training data I'd need. I 
think his solution would have worked if you used a few hundred million positions. I think he used 10,000 games, which
was not nearly enough to be remotely useful. This naivety on my part, meant that some of the options I tried might have
worked had I simply used more data to train.

Predicting the centipawns evaluation worked so poorly in his example, that I decided that I would make my engine by 
predicting the best move. However, I was still learning how to optimally encode inputs and outputs. My first model gave 
two predictions: the square where a move should start and the square where the move should end. This was stupid.

Of course, this meant that most of the moves that it predicted weren't legal moves. I changed it to give me the
probability of where the move would start and end, and then through a bunch of coding acrobatics divined the
best move. Or at least, a move that was legal that was pretty bad.

What's more, I actually had separated predictions for black and white into two different models depending on
whose turn it was. Later, I would learn by studying an Alpha Zero implementation 
(https://github.com/Zeta36/chess-alpha-zero) that there is a clever way of inverting the position so that all 
positions look like it is white-to-move. Technically, the game the model is learning isn't exactly chess, but 
it works, and then you don't need different models for white and black, plus you get double the training data. 
That same Alpha Zero implementation made me realize that I could predict exact moves rather than starting and 
ending locations of pieces, which was a game changer.

My first and second attempts didn't consider time limits, simply looked for moves until it had checked everything
it set out to check. The final (the point where I'm stopping at least for now) engine splits up the searches so 
that it can give a result even after a very short time limit.

I went through a lot of iterations on what the resulting model looked like to augment move predictions. The one that 
I settled on was predicting whether a position was "good", "even", or "bad". I found that predicting exact centipawns
was not as easy as simply checking whether a position was winning. I think that if I kept going, I would probably 
get rid of the "even" evaluation and just lump those positions in with "good".

There were many, many evolutions on how I tried to improve the features and how I tried to improve the actual
move that we made, but my rough conclusion was this: The primary goal was to predict the best move, and 
then verify that the resulting position was good. If the position wasn't good, check the second-best move, and so
on. If we checked all the moves and the positions are all bad, then just use the first predicted move since it
is likely best.

I find it likely that there are more efficient or better ways of doing this, but the end result met my goals.
However, there is one drawback: it takes a lot of data to train the model accurately. Two million unique positions 
was enough for the model to frequently predict good moves, but you can't win a game of chess by getting seven out
of ten moves right. If you have one move that is bad, you will likely lose the game. 

As a human, you quickly learn that throwing away your queen in most situations is bad. As an AI, it has seen situations
where this worked, and it's not afraid to do it, even if this isn't the right situation. This means that sometimes the
best move, the second-best move, and the third-best moves are all terrible. 

Being able to identify whether a position is good or bad helps, but it's really complicated. It took around 4 million 
positions before the evaluation of positions, even just a simple "Is it good or bad?", was usually right. And if we 
didn't want to search through all possible moves (something that is slow to do), it wasn't going to be enough. Remember,
even one bad prediction could lead to losing a game. False positives are the bane of my existence.

I think one of the most valuable lessons that I learned in this process is just how hard it really is to demonstrate
whether a position is winning or not. There are some really clear-cut cases where one side has more material, but
there are just as many cases where the player with more material is about to lose the game to mate in three moves. 
You can't judge a position based on material alone, and if you did, you'd be wrong most of the time! This was a shock
to me, since I thought material advantage (now or even by looking at future moves) would be the primary indicator of 
who is winning. However, when both players play optimally in the early game, material would stay equal. Deciding to 
give up material willingly for a future material gain is complicated. This is where a heuristic evaluation of a 
position can figure it out through thorough investigation, but achieving the same without looking deeper is non-trivial.
However, grandmaster players do something like this. They have seen positions before that they know are winning or
losing and recognize it quickly without an exhaustive consideration of all possibilities. That's what we are trying
to achieve here. This isn't because grandmasters have better intuition than the average player -- it's been proven
that grandmasters are just at bad as everybody else when they see a position that they've never seen before. However,
they have seen a lot of similar positions over thousands of games, and they don't need to calculate to know the likely
best moves. They will check their moves, but they don't look at all possible moves, only those that are likely to
be good. That's what we are trying to do here.

I put a lot of notes in the code as I learned what worked and what didn't. Some of my takeaways are likely wrong
because of other issues that I didn't identify, but I've left everything as-is. 

## What would I do differently?
If I started from scratch right now, knowing what I know now, what would I do differently?

I wasted a lot of time with small issues in my model definition because I didn't understand the impact of all
the parameters or how everything tied together. For example, using bias on the convolutional layers wasn't useful.

I would improve the `MultiFileEvalGenerator` code that feeds in training data. The quality of the results went up
when I had clean, small random batches. The current implementation looks so hacky because of how the code evolved over 
time. I'd rewrite it from scratch to be more elegant and do a better job of creating random batches. Also, take 
responsibility for only including the right amount of data, rather than relying on the processes generating the 
data to right-size it. (Currently, each file has to have exactly 1024 records. I think it would be simpler to 
have the generator trim off the extra records that don't fit cleanly into a batch.)

I'd change the position eval output good or bad (0 or 1) rather than an array representing good, even, or bad. I already 
heuristically detect wins, losses, and draws. There was no need for detecting even, and if there is no need for that,
then there was no need for three options.

The current board state tree is pointless because it doesn't work well with a depth greater than 1. You could simplify
the entire code base to a simple prediction of the top 2 moves, and then take the first one that didn't result in a bad
position or default back to the top predicted move. I'm not sure if what I have is fixable, because I'm not sure if 
I understand why looking deeper doesn't work. My intuition is that a single wrong prediction is really disruptive
and the more predictions we make, the worse the results. So, I'd probably simplify that code and improve the overall
performance. You might think: Why not look at the top 5 or 10 predicted moves and compare them against the prediction 
of the top centipawns for each resulting position? The more predictions you make, the more likely you'll make a bad
prediction, and the wrong move will be chosen. I found after extensive testing, that you are better off taking 1 or
maybe 2 predicted moves and comparing them. Heck, you could remove the validation prediction on whether a board was
good or bad and just take the top predicted move, and you'd likely get as good or better results in most games. So,
why did you leave in the eval check? Mostly because I wanted to see if I could make it useful. With enough training
it eventually became good enough to help slightly more than it hurt.

I'd add support for concurrent predictions. I didn't want to get fancy with this until after I knew I had working 
results. Then I got it working and realized that it wasn't worth putting more time into optimizing something that
nobody was going to want to use.

So much of the code is set up to optimized for handling changes to other previous steps without wasting time, but
we know what we want now. I'd simplify, combine, and reduce how the entire training process worked:
* In one executable, manage the full lifecycle of going from no training data to a fully trained engine
* Instead of saving many pgns, go directly to saving the binary format I need since we know what we need now

And the biggest thing I would change is to not use Stockfish for training. Right now, the entirety of this engine 
relies on Stockfish to evaluate a position for being good or bad and tell us what move is best. I think that if I 
were to continue working on this project, I'd spend the time to build my own heuristic algorithm to make the 
determination. Not because I could do better than Stockfish has done. I sincerely doubt that I could. However, if 
I did, the quality of the results would be completely my own. In building the heuristic aspects, I would learn
something new that I didn't know before. Why bother? If I made a heuristic engine, wouldn't it always give me better
results than ML model that was based on it? Well, this is part of why I'm stopping here. This entire approach isn't
about making the best chess engine possible, but about learning. If I wanted to dive deeper into building a heuristic
chess engine, there really isn't much value in throwing an ML model over it that is less accurate.

Maybe, if I were do another iteration on this, I would take a completely different approach and do more with
self-play. Maybe try to come up with my own approach similar to Alpha Zero and Leela. If I did, though, I probably
wouldn't use Python. Python is slow. Leela needed to play something like 500 million games to play roughly as
well as Stockfish. On my computer with Python, that would likely take years. I don't see much chance of me doing this
since it is a lot of time and effort to recreate something that's been done and I don't feel like I'd learn enough. I'd
have to follow the known recipe to get good results, and at that point, I'm just showing that I know how to read. I
don't think I'm smart enough to innovate a new way of doing it and, for now, reading about it is enough for my own 
purposes.

## Known Issues
While the chess engine supports any arbitrary depth, it really doesn't give good results beyond a depth of 1. There
must be either an issue with how I calculate the best moves or the fact that bad predictions get multiplicatively
worse as you look further into the future. I really didn't put enough time into trying to figure out why it wasn't
working because I didn't need to in order to get my target results. If I wanted this engine to be better, then
this is a problem that would definitely need to be solved.

I don't support the full UCI protocol, and you can't adjust the engine strength. This means that you get what you get.

I've constantly tested the engine while it is playing white. I have barely tested it using black. It might
be fine. It might have issues playing as black that I didn't see because of my lack of testing. The best that I can say
is that when playing against itself, it doesn't seem to break when playing as black.

The engine might make 90% of the best moves possible and then make a move that somebody at 300 elo would agree is bad.
This huge discrepancy in the quality of moves feels bad, but I feel it comes from striving to only find the very best
move rather than to do a search based on evaluating the strength of every position and using the best position. I think 
that this is potentially fixable by splitting the model into two models, one to find the best move and one to find
the best positional evaluation. I did spend weeks trying to give a centipawn evaluation between -15300 and +15300, but
the results were suboptimal. Of course, at the time I had less than 2 million positions to train on. It wasn't until 
I had 10 million positions that I could have used the simple evaluations reliably enough to potentially play a game
with even a low elo opponent.

I only used 23 residual skip blocks because that's all that my video card could hold in memory. I'm pretty sure that 
30 or 40 would have been better. In my experiments, more was always getting better results. There might be an upper
limit where that was no longer true, but I'll never know.

I only used 20 million positions to train on for my final round. I am pretty sure that the results would have been 
even better with 50 million positions or 100 million positions. I needed to pick a stopping point where I had learned
all that I was going to learn from this experience, and we crossed the threshold of diminishing returns.

My training data was based on results from Stockfish with only a depth of 8 moves. I imagine that a depth of 14 or
more would have given even better results. The problem with scoring 20 million positions is that it takes days even 
at a depth of 8. A depth of 14 is far more than twice as slow. If you had a lot of patience, you probably
could improve the final results.

This approach will never be the best approach. The LEG engine can't possibly be better than Stockfish. For one, it 
is learning from Stockfish, but even if it was learning from another chess engine, it's relying on another engine
to tell it what moves are good and what positions are good. This is completely different from Alpha Zero where it 
figures out what is good or bad on its own. When I saw the code from Alpha Zero, I knew that I could potentially
get it running on my machine and train it, but then what? I didn't build something new or learn anything. This was
about learning after all, so I didn't seek to emulate its approach of self-learning.

I didn't finish creating support for doing concurrent predictions.

# Training Approach
I downloaded tens of thousands of PGNs from http://www.pgnmentor.com/files.html as my starting point, but this 
was completely pointless. Because my original plan was to teach the engine the best moves based on who won a game
and I only wanted to use good games, this seemed like a good start at the time. It wasn't. I quickly realized that
that approach wasn't going to give me enough data and what's more, even great players make mistakes.

So, I made a script to generate PGNs by playing Stockfish against itself for thousands of games at differing strength.
This helped me get to tens of thousands, maybe even a million positions, but it was super slow-going. What's more, 
the LEG engine had lots of blind spots. 

I made two huge advancements that helped improve training. First, I set it up to play against itself using random
openings. This would lead to tens of thousands of new positions after each training session. And secondly, I created
a script that would look at every game in my training data and then generate new training data for all possible moves 
that we hadn't already considered. Filling in those positions that were similar, but with drastically different
outcomes was probably the single best thing I did this project.

I should point out that I had found that it was very important not to have any positions duplicated in my training data.
There was probably three of four weeks early on that I didn't realize this, and it skewed my results heavily since some
positions were heavily over-represented. Think about all the openings that are played repeatedly. The value of specific
moves are going to be way more common and that will lead to the prediction of those moves.


# Installing
## Where to get a free Chess GUI

Personally, I installed Lucas Chess:
https://lucaschess.pythonanywhere.com/downloads (Source code: https://github.com/lukasmonk/lucaschessR)

This works great when pointed at the chess engine in the dist folder, but I had issues with it working when
installed in the program files(x86) folder, even though Shredder can use the LEG engine from that folder fine.

Other options (which I haven't tried):
* The Tarrasch Chess GUI: https://www.triplehappy.com/downloads.html
* ChessBase Reader: https://en.chessbase.com/pages/download (windows only)
* Arena: http://www.playwitharena.de/ (windows only)

## How to watch engine-vs-engine matches

While you can watch text-only games fly by using the scripts that I provided, if you have want to watch
in luxury, you have options. I personally use Deep Shredder 13:
https://www.shredderchess.com/windows/deep-shredder-13.html

If you want a free option, and you are on Windows, I have read that Arena works well: http://www.playwitharena.de/

## Referencing the UCI engine

If you installed from source and have python in your path, you can reference the engine in a chess GUI like this:
```commandline
python.exe leg_uci_engine.py
```

If you want to reference it through code, it looks like this:
```python
engine = chess.engine.SimpleEngine.popen_uci([sys.executable, r"leg_uci_engine.py"])
```

## Pre-packaged LEG engine
The Output folder has a `legchess_setup.exe` which you can run to install the LEG Chess Engine on your computer.

If you place it in program files, you can then point at it using a chess engine like Shredder.

The prepackaged engine doesn't use an opening book or endgame tables (and it doesn't really seem to impact the win 
rate much) since it inflates the overall install size to more than 64 gigabytes.

If you want it to play reasonable openings, you should have your chess GUI use an opening book
on behalf of the engine.

## LEG engine From Source
I'm skeptical that another human being will ever download the source and actually get it running exactly the way
that I did on my machine, but here's to hoping! This whole process was a learning experience for me and not designed
as something I intended others to actually use. That said, I did imagine that others might be able to learn from it 
and create their own code.

Because many of the files are very big and GitHub gets grumpy about it, you need to install the python code first
and then install the other supporting files (opening books, endgame tables, training engine, and model) separately.

If you've downloaded the LEG Engine from GitHub, you can install the requirements with:
```
pip install -r requirements.txt
```
As I recall, I had to install drivers for my video card to support TensorFlow and Cuda. I don't remember at the moment
what those drivers were, but I remember using a combination of Google and reading the error messages.

In the `leg_uci_engine.py` source file, the engine is configured to use the default model that I last trained with,
and has opening books and endgame tables disabled. Be sure to update this code to reflect what you have:
```python
engine = LegEngine(ai_model_file=DEFAULT_AI_MODEL, opening_book_file_path=None, use_endgame_tables=False)
```

If you don't use the prepackaged exe, you'll need to rebundle into an exe after training and including an opening
book and endgame tables:
```
# basic
pyinstaller --noconfirm --add-data models;models leg_uci_engine.py 


# If you want to distribute the opening book
pyinstaller --noconfirm --add-data models;models --add-data openingbooks/Titans.bin;openingbooks/Titans.bin leg_uci_engine.py 

# If you enable endgames and opening book in the code, you'll have to update the leg_uci_engine.py file and then add them to the package:
pyinstaller --noconfirm --add-data models;models --add-data openingbooks/Titans.bin;openingbooks/Titans.bin --add-data endgames/6-WDL;endgames/6-WDL leg_uci_engine.py 

```
#### Using pre-trained model 
If you want to use my model, you'll need to add a folder named "models":
```commandline
mkdir models
```

And you'll need to download and place my model file in that new folder.

#### Adding endgame tables
**At the time I wrote this, end game files need to be in the folder**:
`endgames/6-WDL`

One could change the folder the code is looking at by modifying `find_end_game_move(current_board)` in
`data_utils.py`.

I put details on where I got my endgame tables here: `endgames/file_sources.txt`

I also made a utility to download them: `download_endgame.py`

Please notice how large the 7-move endgame files are. You should not attempt to download them unless you are confident
that you have enough disk and are willing to wait the time it takes to download them.

#### Adding opening books

I provide locations on how I got opening books here: `openingbooks/polyglot.txt`

#### Where to get Stockfish
If you decide to train your own model, you'll probably want stockfish:
https://stockfishchess.org/download/

While you could train using Shredder or another chess engine, I didn't try it, so I'm not sure what would happen.

## What will I do next?
I have a lot on my plate. Outside my day job as a Solutions Architect, I have a couple of science fiction books 
in the works and a 2d fantasy RPG that I'm doing everything for (including all the art, which is very intimidating.)

There's very little chance that I'll revisit this project or improve upon it.

