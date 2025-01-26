defmodule Scholar.Preprocessing.BinarizerTest do
  use Scholar.Case, async: true
  alias Scholar.FeatureExtraction.CountVectorizer
  doctest CountVectorizer

  describe "fit_transform" do
    test "dupa" do
      require Explorer.DataFrame, as: DF
      require Explorer.Series, as: S
      Nx.global_default_backend(EXLA.Backend)
      # Client can also be set to :cuda / :rocm
      Nx.Defn.global_default_options(compiler: EXLA, client: :host)

      df =
        DF.from_csv!("/Users/karol/Inz/scholar/test/scholar/feature_extraction/dupa2.csv")

      text = df[:text]
      # 1 - spam, 0 - ham
      y = df[:label_num]
      {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-cased")

      # df =
      #   df
      #   |> DF.put(
      #     :text,
      #     S.transform(df[:text], fn n ->
      #       result = CountVectorizer.fit_transform([n])

      #       Nx.to_list(result.counts)
      #     end)
      #   )

      text = df[:text] |> S.to_list() |> CountVectorizer.fit_transform()
      text = text.counts

      train_ratio = 0.8
      {x_train, x_test} = Nx.split(text, train_ratio)
      {y_train, y_test} = Nx.split(y, train_ratio)
      bnb = Scholar.NaiveBayes.Bernoulli.fit(x_train, y_train, num_classes: 2)
      y_pred = Scholar.NaiveBayes.Bernoulli.predict(bnb, x_test, Nx.tensor([0, 1]))

      Scholar.Metrics.Classification.f1_score(
        y_test,
        y_pred,
        num_classes: 2,
        average: :macro
      )
      |> IO.inspect()

      histogram = Tucan.histogram(:cars, "Horsepower", color_by: "Origin", fill_opacity: 0.5)
    end

    test "fit_transform test - default options" do
      result = CountVectorizer.fit_transform(["i love elixir", "hello world"])

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 1]
        ])

      expected_vocabulary = %{
        "elixir" => Nx.tensor(0),
        "hello" => Nx.tensor(1),
        "i" => Nx.tensor(2),
        "love" => Nx.tensor(3),
        "world" => Nx.tensor(4)
      }

      assert result.counts == expected_counts
      assert result.vocabulary == expected_vocabulary
    end

    test "fit_transform test - removes interpunction" do
      result = CountVectorizer.fit_transform(["i love elixir.", "hello, world!"])

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 1]
        ])

      expected_vocabulary = %{
        "elixir" => Nx.tensor(0),
        "hello" => Nx.tensor(1),
        "i" => Nx.tensor(2),
        "love" => Nx.tensor(3),
        "world" => Nx.tensor(4)
      }

      assert result.counts == expected_counts
      assert result.vocabulary == expected_vocabulary
    end

    test "fit_transform test - ignores case" do
      result = CountVectorizer.fit_transform(["i love elixir", "hello world HELLO"])

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 2, 0, 0, 1]
        ])

      expected_vocabulary = %{
        "elixir" => Nx.tensor(0),
        "hello" => Nx.tensor(1),
        "i" => Nx.tensor(2),
        "love" => Nx.tensor(3),
        "world" => Nx.tensor(4)
      }

      assert result.counts == expected_counts
      assert result.vocabulary == expected_vocabulary
    end

    test "fit_transform test - already indexed tensor" do
      result =
        CountVectorizer.fit_transform(
          Nx.tensor([
            [2, 3, 0],
            [1, 4, 4]
          ]),
          indexed_tensor: true
        )

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 2]
        ])

      assert result.counts == expected_counts
      assert result.vocabulary == %{}
    end

    test "fit_transform test - already indexed tensor with padding" do
      result =
        CountVectorizer.fit_transform(
          Nx.tensor([
            [2, 3, 0],
            [1, 4, -1]
          ]),
          indexed_tensor: true
        )

      expected_counts =
        Nx.tensor([
          [1, 0, 1, 1, 0],
          [0, 1, 0, 0, 1]
        ])

      assert result.counts == expected_counts
      assert result.vocabulary == %{}
    end
  end
end
