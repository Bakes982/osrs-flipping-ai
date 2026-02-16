package com.flippingai;

import java.util.List;

public class OpportunityList
{
    public List<Opportunity> items;
    public int total;

    public static class Opportunity
    {
        public int item_id;
        public String name;
        public long buy_price;
        public long sell_price;
        public long margin;
        public double roi_pct;
        public int volume;
        public long potential_profit;
        public double ml_confidence;

        public String getFormattedProfit()
        {
            if (potential_profit >= 1_000_000)
            {
                return String.format("%.1fM", potential_profit / 1_000_000.0);
            }
            if (potential_profit >= 1_000)
            {
                return String.format("%.1fK", potential_profit / 1_000.0);
            }
            return String.valueOf(potential_profit);
        }
    }
}
