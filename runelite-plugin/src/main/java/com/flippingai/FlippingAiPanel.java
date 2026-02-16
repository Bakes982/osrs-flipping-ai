package com.flippingai;

import lombok.extern.slf4j.Slf4j;
import net.runelite.client.ui.ColorScheme;
import net.runelite.client.ui.PluginPanel;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.text.NumberFormat;
import java.util.List;

@Slf4j
public class FlippingAiPanel extends PluginPanel
{
    private static final Color COLOR_BUY = new Color(6, 182, 212);
    private static final Color COLOR_SELL = new Color(249, 115, 22);
    private static final Color COLOR_PROFIT = new Color(16, 185, 129);
    private static final Color COLOR_BG_DARK = new Color(15, 23, 42);
    private static final Color COLOR_BG_CARD = new Color(30, 41, 59);
    private static final Color COLOR_BORDER = new Color(51, 65, 85);

    private final FlippingAiPlugin plugin;
    private final JPanel opportunityList;
    private final JLabel statusLabel;
    private final NumberFormat nf = NumberFormat.getInstance();

    public FlippingAiPanel(FlippingAiPlugin plugin)
    {
        super(false);
        this.plugin = plugin;

        setBackground(COLOR_BG_DARK);
        setLayout(new BorderLayout());

        // Header
        JPanel header = new JPanel(new BorderLayout());
        header.setBackground(COLOR_BG_DARK);
        header.setBorder(new EmptyBorder(10, 10, 5, 10));

        JLabel title = new JLabel("Flipping AI");
        title.setFont(title.getFont().deriveFont(Font.BOLD, 16f));
        title.setForeground(Color.WHITE);
        header.add(title, BorderLayout.WEST);

        statusLabel = new JLabel("Connecting...");
        statusLabel.setForeground(new Color(148, 163, 184));
        statusLabel.setFont(statusLabel.getFont().deriveFont(11f));
        header.add(statusLabel, BorderLayout.EAST);

        add(header, BorderLayout.NORTH);

        // Scrollable opportunity list
        opportunityList = new JPanel();
        opportunityList.setLayout(new BoxLayout(opportunityList, BoxLayout.Y_AXIS));
        opportunityList.setBackground(COLOR_BG_DARK);

        JScrollPane scrollPane = new JScrollPane(opportunityList);
        scrollPane.setBackground(COLOR_BG_DARK);
        scrollPane.getViewport().setBackground(COLOR_BG_DARK);
        scrollPane.setBorder(null);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);

        add(scrollPane, BorderLayout.CENTER);

        // Initial placeholder
        addPlaceholder();
    }

    public void updateOpportunities(OpportunityList opps)
    {
        SwingUtilities.invokeLater(() -> {
            opportunityList.removeAll();

            if (opps == null || opps.items == null || opps.items.isEmpty())
            {
                addPlaceholder();
                statusLabel.setText("No data");
                return;
            }

            statusLabel.setText(opps.items.size() + " opportunities");

            for (OpportunityList.Opportunity opp : opps.items)
            {
                opportunityList.add(createOpportunityCard(opp));
                opportunityList.add(Box.createRigidArea(new Dimension(0, 4)));
            }

            opportunityList.revalidate();
            opportunityList.repaint();
        });
    }

    private JPanel createOpportunityCard(OpportunityList.Opportunity opp)
    {
        JPanel card = new JPanel();
        card.setLayout(new BoxLayout(card, BoxLayout.Y_AXIS));
        card.setBackground(COLOR_BG_CARD);
        card.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(COLOR_BORDER, 1),
            new EmptyBorder(8, 10, 8, 10)
        ));
        card.setMaximumSize(new Dimension(Integer.MAX_VALUE, 120));
        card.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Item name row
        JPanel nameRow = new JPanel(new BorderLayout());
        nameRow.setBackground(COLOR_BG_CARD);
        nameRow.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel nameLabel = new JLabel(opp.name != null ? opp.name : "Item #" + opp.item_id);
        nameLabel.setForeground(Color.WHITE);
        nameLabel.setFont(nameLabel.getFont().deriveFont(Font.BOLD, 12f));
        nameRow.add(nameLabel, BorderLayout.WEST);

        JLabel profitLabel = new JLabel("+" + opp.getFormattedProfit() + " GP");
        profitLabel.setForeground(COLOR_PROFIT);
        profitLabel.setFont(profitLabel.getFont().deriveFont(Font.BOLD, 12f));
        nameRow.add(profitLabel, BorderLayout.EAST);

        card.add(nameRow);
        card.add(Box.createRigidArea(new Dimension(0, 4)));

        // Price row
        JPanel priceRow = new JPanel(new BorderLayout());
        priceRow.setBackground(COLOR_BG_CARD);
        priceRow.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel buyLabel = new JLabel("Buy: " + nf.format(opp.buy_price));
        buyLabel.setForeground(COLOR_BUY);
        buyLabel.setFont(buyLabel.getFont().deriveFont(11f));
        priceRow.add(buyLabel, BorderLayout.WEST);

        JLabel sellLabel = new JLabel("Sell: " + nf.format(opp.sell_price));
        sellLabel.setForeground(COLOR_SELL);
        sellLabel.setFont(sellLabel.getFont().deriveFont(11f));
        priceRow.add(sellLabel, BorderLayout.EAST);

        card.add(priceRow);
        card.add(Box.createRigidArea(new Dimension(0, 2)));

        // Stats row
        JPanel statsRow = new JPanel(new BorderLayout());
        statsRow.setBackground(COLOR_BG_CARD);
        statsRow.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel roiLabel = new JLabel(String.format("ROI: %.1f%%", opp.roi_pct));
        roiLabel.setForeground(new Color(148, 163, 184));
        roiLabel.setFont(roiLabel.getFont().deriveFont(10f));
        statsRow.add(roiLabel, BorderLayout.WEST);

        JLabel volLabel = new JLabel("Vol: " + nf.format(opp.volume));
        volLabel.setForeground(new Color(148, 163, 184));
        volLabel.setFont(volLabel.getFont().deriveFont(10f));
        statsRow.add(volLabel, BorderLayout.EAST);

        card.add(statsRow);

        // Click to request prediction
        card.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        card.addMouseListener(new java.awt.event.MouseAdapter()
        {
            @Override
            public void mouseClicked(java.awt.event.MouseEvent e)
            {
                plugin.requestPrediction(opp.item_id);
            }

            @Override
            public void mouseEntered(java.awt.event.MouseEvent e)
            {
                card.setBackground(new Color(51, 65, 85));
                for (Component c : card.getComponents())
                {
                    if (c instanceof JPanel) c.setBackground(new Color(51, 65, 85));
                }
            }

            @Override
            public void mouseExited(java.awt.event.MouseEvent e)
            {
                card.setBackground(COLOR_BG_CARD);
                for (Component c : card.getComponents())
                {
                    if (c instanceof JPanel) c.setBackground(COLOR_BG_CARD);
                }
            }
        });

        return card;
    }

    private void addPlaceholder()
    {
        JLabel placeholder = new JLabel("Waiting for backend...");
        placeholder.setForeground(new Color(100, 116, 139));
        placeholder.setHorizontalAlignment(SwingConstants.CENTER);
        placeholder.setBorder(new EmptyBorder(40, 10, 10, 10));
        placeholder.setAlignmentX(Component.CENTER_ALIGNMENT);
        opportunityList.add(placeholder);
    }
}
