open(START, $ARGV[0]);
while ($line=<START>) {
	print $line if ($line =~ /\\newcommand/);
	if ($line =~ /\\input\{.*\}/) {
	  findnc($1);
      }
}
close START;

sub findnc {
#	my IN=undef;
open(IN, $_);
while ($line=<IN>) {
	print $line if ($line =~ /\\newcommand/);
}
close IN;
}
