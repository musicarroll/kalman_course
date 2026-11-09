open(GS,"gs.bat");
while ($cmd=<GS>) {
	print "Executing: $cmd";
	system($cmd);
}
close GS;
