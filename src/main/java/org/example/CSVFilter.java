package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class CSVFilter {
    public static void main(String[] args) {
        String inputFilePath = "src/main/data_resources/SPX.csv";  // Correct path to the input CSV
        String outputFilePath = "src/main/data_resources/TrimmedSPX.csv";  // Output file
        LocalDate cutoffDate = LocalDate.of(2010, 1, 1);  // Filtering cutoff date
        DateTimeFormatter dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");

        try (Reader reader = Files.newBufferedReader(Paths.get(inputFilePath));
             Writer writer = Files.newBufferedWriter(Paths.get(outputFilePath));
             CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"))) {

            Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);

            for (CSVRecord record : records) {
                // Parse the date from the current record
                LocalDate date = LocalDate.parse(record.get("Date"), dateFormatter);

                // If the date is after or equal to the cutoff date, print the record to the new CSV
                if (date.isAfter(cutoffDate) || date.isEqual(cutoffDate)) {
                    csvPrinter.printRecord(record);
                }
            }

            System.out.println("Filtered data saved to " + outputFilePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
