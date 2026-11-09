collatz_sequence(N) :-
    collatz_sequence(N, 0).

collatz_sequence(1, Level) :-
    tab(Level), write(1), nl.
collatz_sequence(N, Level) :-
    tab(Level), write(N), nl,
    (   N mod 2 =:= 0 ->
        Next is N // 2
    ;   Next is 3 * N + 1
    ),
    NextLevel is Level + 1,
    collatz_sequence(Next, NextLevel).

% Example usage
:- initialization(main).
main :-
    write('Enter a positive integer: '),
    read(Number),
    write('Collatz sequence:'), nl,
    collatz_sequence(Number),
    halt.
