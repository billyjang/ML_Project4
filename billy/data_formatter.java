import java.util.*;
import java.io.*;

public class data_formatter {
    public static void main(String[] args) {
        try {
            Scanner infile = new Scanner(new FileReader("first_full_predictions.csv"));
            ArrayList<String> lines = new ArrayList<String>();
            while(infile.hasNextLine()) {
                lines.add(infile.nextLine());
            }
            System.out.println(lines.size());
            PrintWriter printer = new PrintWriter("first_full_formatted.csv");
            for(int i = 0; i < lines.size(); i++) {
                String s = "";
                String line = lines.get(i);
                String[] parts = line.split(" ");
                Double double_id = Double.parseDouble(parts[0]);
                int int_id = (int)Math.round(double_id);
                s = int_id + "," + parts[1] + "," + parts[2] + "\n";
                printer.print(s);
                System.out.println(i);
            }
            infile.close();
            printer.close();
        } catch(FileNotFoundException ex) {

        }

    }
}