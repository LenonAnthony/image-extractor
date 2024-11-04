const { z } = require("zod");

const TextExtract = z.object({
  title: z.string().describe("The perceived title on the image"),
  main_text: z.string().describe("The main text on the file"),
  objects_in_image: z
    .string()
    .describe("Any other objects observed in the image"),
});

const TextExtractWithImage = z.object({
  path: z.string().describe("The original path of the image"),
});

module.exports = {
  TextExtract,
  TextExtractWithImage,
};
