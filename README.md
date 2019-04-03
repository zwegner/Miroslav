Miroslav
========

Miroslav is a fast regular expression matching algorithm using SIMD instructions to do a lossy NFA simulation on many input bytes in parallel.

It's named after Miroslav Vitous, since his song [Infinite Search](https://www.youtube.com/watch?v=-OdIEbFwQEs) (from the album Infinite Search) came on shuffle as I was writing the initial implementation, and that seemed like an appropriate fit. The whole album is amazing by the way, some great early electric jazz from 1970.

A full writeup on the algorithm should be coming soon-ish.

This implementation for now only has a few features:

* regular expressions can consist only of literal strings and the alternation operator `|`.
* patterns can be loaded from a file with `-f [file]`
* the count of lines matched can be printed with `-c`, otherwise all matching lines are printed
