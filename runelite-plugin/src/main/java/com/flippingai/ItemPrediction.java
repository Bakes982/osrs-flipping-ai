package com.flippingai;

import java.util.HashMap;
import java.util.Map;

public class ItemPrediction
{
    public int itemId;
    public String itemName;
    public long currentBuy;
    public long currentSell;
    public long suggestedBuy;
    public long suggestedSell;
    public long expectedProfit;
    public double confidence;
    public String horizon;
    public Map<String, String> directions = new HashMap<>();
    public Map<String, Double> confidences = new HashMap<>();

    public String getFormattedProfit()
    {
        if (expectedProfit >= 1_000_000)
        {
            return String.format("%.1fM", expectedProfit / 1_000_000.0);
        }
        if (expectedProfit >= 1_000)
        {
            return String.format("%.1fK", expectedProfit / 1_000.0);
        }
        return String.valueOf(expectedProfit);
    }

    public String getConfidenceColor()
    {
        if (confidence >= 0.7) return "00ff00"; // green
        if (confidence >= 0.5) return "ffff00"; // yellow
        return "ff6600"; // orange
    }
}
