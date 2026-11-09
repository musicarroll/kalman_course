$DEBUG = 1;
open(IN,$ARGV[0]);
while ($file=<IN>) {
  chomp $file;
  ($pre,$post) = split(/c130_/,$file);
  print "\"$pre\"\t\"$post\"\n" if ($DEBUG);
  $newfile = "gps_" . $post;
  system("mv $file $newfile");
}
close IN;
