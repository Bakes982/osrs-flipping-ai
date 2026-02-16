package com.flippingai;

import net.runelite.api.Client;
import net.runelite.api.widgets.Widget;
import net.runelite.client.ui.overlay.Overlay;
import net.runelite.client.ui.overlay.OverlayLayer;
import net.runelite.client.ui.overlay.OverlayPosition;
import net.runelite.client.ui.overlay.OverlayPriority;
import net.runelite.client.ui.overlay.components.LineComponent;
import net.runelite.client.ui.overlay.components.PanelComponent;
import net.runelite.client.ui.overlay.components.TitleComponent;

import javax.inject.Inject;
import java.awt.*;
import java.text.NumberFormat;

public class FlippingAiOverlay extends Overlay
{
    private static final Color COLOR_BUY = new Color(6, 182, 212);   // cyan
    private static final Color COLOR_SELL = new Color(249, 115, 22); // orange
    private static final Color COLOR_PROFIT = new Color(16, 185, 129); // green
    private static final Color COLOR_HEADER = new Color(148, 163, 184); // slate
    private static final Color COLOR_BG = new Color(15, 23, 42, 220);  // dark navy

    private static final int GE_OFFER_WINDOW_GROUP = 465;

    private final Client client;
    private final FlippingAiPlugin plugin;
    private final FlippingAiConfig config;
    private final PanelComponent panelComponent = new PanelComponent();
    private final NumberFormat nf = NumberFormat.getInstance();

    @Inject
    public FlippingAiOverlay(Client client, FlippingAiPlugin plugin, FlippingAiConfig config)
    {
        super(plugin);
        this.client = client;
        this.plugin = plugin;
        this.config = config;

        setPosition(OverlayPosition.TOP_CENTER);
        setLayer(OverlayLayer.ABOVE_WIDGETS);
        setPriority(OverlayPriority.HIGH);
    }

    @Override
    public Dimension render(Graphics2D graphics)
    {
        if (!config.showOverlay())
        {
            return null;
        }

        // Only show when GE offer window is open
        Widget geWidget = client.getWidget(GE_OFFER_WINDOW_GROUP, 0);
        if (geWidget == null || geWidget.isHidden())
        {
            return null;
        }

        ItemPrediction prediction = plugin.getCurrentPrediction();
        if (prediction == null || prediction.suggestedBuy == 0)
        {
            return null;
        }

        panelComponent.getChildren().clear();
        panelComponent.setBackgroundColor(COLOR_BG);
        panelComponent.setPreferredSize(new Dimension(220, 0));

        // Title
        panelComponent.getChildren().add(TitleComponent.builder()
            .text("Flipping AI")
            .color(Color.WHITE)
            .build());

        // Item name
        panelComponent.getChildren().add(LineComponent.builder()
            .left(prediction.itemName)
            .leftColor(Color.WHITE)
            .build());

        // Separator
        panelComponent.getChildren().add(LineComponent.builder()
            .left("───────────────────")
            .leftColor(COLOR_HEADER)
            .build());

        // Buy suggestion
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Buy at:")
            .leftColor(COLOR_HEADER)
            .right(nf.format(prediction.suggestedBuy) + " GP")
            .rightColor(COLOR_BUY)
            .build());

        // Sell suggestion
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Sell at:")
            .leftColor(COLOR_HEADER)
            .right(nf.format(prediction.suggestedSell) + " GP")
            .rightColor(COLOR_SELL)
            .build());

        // Expected profit
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Exp. Profit:")
            .leftColor(COLOR_HEADER)
            .right(prediction.getFormattedProfit() + " GP")
            .rightColor(COLOR_PROFIT)
            .build());

        // Horizon
        if (prediction.horizon != null && !prediction.horizon.isEmpty())
        {
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Horizon:")
                .leftColor(COLOR_HEADER)
                .right(prediction.horizon)
                .rightColor(Color.WHITE)
                .build());
        }

        // Confidence
        if (config.showConfidence())
        {
            int confPct = (int) (prediction.confidence * 100);
            Color confColor = confPct >= 70 ? COLOR_PROFIT :
                confPct >= 50 ? Color.YELLOW : COLOR_SELL;

            panelComponent.getChildren().add(LineComponent.builder()
                .left("Confidence:")
                .leftColor(COLOR_HEADER)
                .right(confPct + "%")
                .rightColor(confColor)
                .build());
        }

        // Trend direction for selected horizon
        String dir = prediction.directions.get(config.horizon().getApiKey());
        if (dir != null)
        {
            String arrow = "up".equals(dir) ? "▲" : "down".equals(dir) ? "▼" : "►";
            Color dirColor = "up".equals(dir) ? COLOR_PROFIT :
                "down".equals(dir) ? new Color(239, 68, 68) : COLOR_HEADER;

            panelComponent.getChildren().add(LineComponent.builder()
                .left("Trend:")
                .leftColor(COLOR_HEADER)
                .right(arrow + " " + dir.toUpperCase())
                .rightColor(dirColor)
                .build());
        }

        return panelComponent.render(graphics);
    }
}
