
open(IN,$ARGV[0]);
$lastnum=0;
while ($line=<IN>) {
	if ($line =~ /\\onslide\<(\d+)\-\>/) {
		$lastnum = $1;
#		print "$line";
	}
	if ($line =~ /end\{frame\}/) {
#	  print "$lastnum\n";
	  $newnum = $lastnum+1;
	  print "\\onslide\<$newnum\-\>\{\\center --------------------------------------------------------------------\}\n";
	  $lastnum=0;
	}
	print $line;
}
close IN;
